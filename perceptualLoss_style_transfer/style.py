from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import requests
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms

import utils
from network import ImageTransformNet
from vgg import Vgg16

# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7


class COCOFromURLDataset(Dataset):
    def __init__(self, annotation_file, transform=None, preload=False, cache_dir='./cache', num_preload=None):
        """
        Args:
            annotation_file (string): Path to the json file with COCO annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
            preload (bool): Whether to preload images.
            cache_dir (string): Directory to cache preloaded images.
            num_preload (int or None): Number of images to preload. If None, preload all.
        """
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.preload = preload
        self.cache_dir = cache_dir
        self.num_preload = num_preload or len(self.ids)
        
        # Shuffle ids for preloading
        np.random.shuffle(self.ids)
        
        if preload:
            self.preloaded_data = {}
            self._preload_images()

    def _download_image(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img, img_id

    def _preload_images(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Use ThreadPoolExecutor to parallelize downloads
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_id = {executor.submit(self._download_image, img_id): img_id for img_id in self.ids[:self.num_preload]}
            for future in future_to_id:
                img, img_id = future.result()
                cache_path = os.path.join(self.cache_dir, f"{img_id}.jpg")
                img.save(cache_path)
                self.preloaded_data[img_id] = cache_path

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        if self.preload:
            img_path = self.preloaded_data.get(img_id)
            img = Image.open(img_path).convert('RGB')
        else:
            img, _ = self._download_image(img_id)

        if self.transform:
            img = self.transform(img)

        return img, img_id
    
    
def train(args):          
    # GPU enabling
    device = torch.device('cpu')
    if args.gpu is not None:
        device = torch.device("cuda:{}".format(args.gpu))
        use_cuda = True
        # dtype = torch.cuda.FloatTensor
        # torch.cuda.set_device(args.gpu)
        # print("Current device: %d" %torch.cuda.current_device())
        
    # visualization of training controlled by flag
    visualize = (args.visualize != None)
    if (visualize):
        img_transform_512 = transforms.Compose([
            transforms.Resize(512),                  # scale shortest side to image_size
            transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
        ])

        testImage_amber = utils.load_image("content_imgs/amber.jpg")
        testImage_amber = img_transform_512(testImage_amber)
        testImage_amber = Variable(testImage_amber.repeat(1, 1, 1, 1), requires_grad=False).to(device)

        testImage_dan = utils.load_image("content_imgs/dan.jpg")
        testImage_dan = img_transform_512(testImage_dan)
        testImage_dan = Variable(testImage_dan.repeat(1, 1, 1, 1), requires_grad=False).to(device)

        testImage_maine = utils.load_image("content_imgs/maine.jpg")
        testImage_maine = img_transform_512(testImage_maine)
        testImage_maine = Variable(testImage_maine.repeat(1, 1, 1, 1), requires_grad=False).to(device)

    # define network
    image_transformer = ImageTransformNet().to(device)
    optimizer = Adam(image_transformer.parameters(), LEARNING_RATE) 

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().to(device)
    """ Load coco dataset """
    dataset_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(), utils.normalize_tensor_transform()])
    # Initialize dataset
    annotation_file = '/h/u14/c9/00/zhaoha36/Desktop/CSC413/csc413_project/annotations/instances_train2014.json'
    train_dataset = COCOFromURLDataset(annotation_file=annotation_file, transform=dataset_transform)

    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])
    style = utils.load_image(args.style_image)
    style = style_transform(style)
    style = Variable(style.repeat(BATCH_SIZE, 1, 1, 1)).to(device)
    style_name = os.path.split(args.style_image)[-1].split('.')[0]

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]

    for e in range(EPOCHS):

        # track values for...
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, (x, label) in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = Variable(x).to(device)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                style_loss += loss_mse(y_hat_gram[j], style_gram[j][:img_batch_read])
            style_loss = STYLE_WEIGHT*style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                aggregate_style_loss/(batch_num+1.0), aggregate_content_loss/(batch_num+1.0), aggregate_tv_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item()
                            )
                print(status)

            if ((batch_num + 1) % 1000 == 0) and (visualize):
                image_transformer.eval()

                if not os.path.exists("visualization"):
                    os.makedirs("visualization")
                if not os.path.exists("visualization/%s" %style_name):
                    os.makedirs("visualization/%s" %style_name)

                outputTestImage_amber = image_transformer(testImage_amber).cpu()
                amber_path = "visualization/%s/amber_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                utils.save_image(amber_path, outputTestImage_amber.item())

                outputTestImage_dan = image_transformer(testImage_dan).cpu()
                dan_path = "visualization/%s/dan_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                utils.save_image(dan_path, outputTestImage_dan.item())

                outputTestImage_maine = image_transformer(testImage_maine).cpu()
                maine_path = "visualization/%s/maine_%d_%05d.jpg" %(style_name, e+1, batch_num+1)
                utils.save_image(maine_path, outputTestImage_maine.item())

                print("images saved")
                image_transformer.train()

    # save model
    image_transformer.eval()

    if use_cuda:
        image_transformer.cpu()

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    torch.save(image_transformer.state_dict(), filename)
    
    if use_cuda:
        image_transformer.cuda()

def style_transfer(args):
    # GPU enabling
    if (args.gpu != None):
        use_cuda = True
        dtype = torch.cuda.FloatTensor
        torch.cuda.set_device(args.gpu)
        print("Current device: %d" %torch.cuda.current_device())

    # content image
    img_transform_512 = transforms.Compose([
            transforms.Resize(512),                  # scale shortest side to image_size
            transforms.CenterCrop(512),             # crop center image_size out
            transforms.ToTensor(),                  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()      # normalize with ImageNet values
    ])

    content = utils.load_image(args.source)
    content = img_transform_512(content)
    content = content.unsqueeze(0)
    content = Variable(content).type(dtype)

    # load style model
    style_model = ImageTransformNet().type(dtype)
    # style_model.load_state_dict(torch.load(args.model_path))
    
    checkpoint = torch.load(args.model_path)
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if not (k.endswith('.running_mean') or k.endswith('.running_var'))}
    style_model.load_state_dict(filtered_checkpoint)

	# process input image
    stylized = style_model(content).cpu()
    utils.save_image(args.output, stylized.data[0])


def main():
    parser = argparse.ArgumentParser(description='style transfer in pytorch')
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    train_parser = subparsers.add_parser("train", help="train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, required=True, help="path to a style image to train with")
    train_parser.add_argument("--dataset", type=str, required=False, help="path to a dataset, defalut is COCO dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")

    style_parser = subparsers.add_parser("transfer", help="do style transfer with a trained model")
    style_parser.add_argument("--model-path", type=str, required=True, help="path to a pretrained model for a style image")
    style_parser.add_argument("--source", type=str, required=True, help="path to source image")
    style_parser.add_argument("--output", type=str, required=True, help="file name for stylized output image")
    style_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    args = parser.parse_args()

    # command
    if (args.subcommand == "train"):
        print("Training!")
        train(args)
    elif (args.subcommand == "transfer"):
        print("Style transfering!")
        style_transfer(args)
    else:
        print("invalid command")

if __name__ == '__main__':
    main()
    # example usage to perform style transferring on existing trained model:
    # srun -p csc413 --gres gpu python3 style.py transfer --model-path models/mosaic.model --source content_imgs/maine.jpg --gpu 0 --output figure/output_image.png
