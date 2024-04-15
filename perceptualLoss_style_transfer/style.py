from matplotlib import pyplot as plt
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import utils
from network import ImageTransformNet
from vgg import Vgg16
from simple_dataset import SimpleDataset
from PIL import Image


# Global Variables
IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 2
STYLE_WEIGHT = 1e5
CONTENT_WEIGHT = 1e0
TV_WEIGHT = 1e-7
SEED = 1080
torch.manual_seed(SEED)
LOG_INTERVAL = 100  # Adjust to control how often to log

def train(args):          
    if torch.cuda.is_available():
        use_cuda = True
        device = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
        torch.set_default_device(device)
        print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        use_cuda = False
        device = torch.device("cpu")
        torch.set_default_device(device)
        raise SystemError("CUDA is not available. Check your installation.")
            
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
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    DATASET = script_dir + "/../train2014-2"
    dataset_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(), utils.normalize_tensor_transform()])
    # Initialize dataset
    train_dataset = SimpleDataset(DATASET, dataset_transform)
    print(len(train_dataset))

    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

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

    iteration_count = 0

    # Initialization for plotting losses
    iterations = []
    avg_content_losses = []
    avg_style_losses = []
    avg_tv_losses = []
    avg_total_losses = []

    for e in range(EPOCHS):
        agg_content_loss = agg_style_loss = agg_tv_loss = 0.0

        img_count = 0

        # train network
        image_transformer.train()
        for batch_num, x in enumerate(train_loader):
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

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]      
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT*loss_mse(recon_hat, recon)

            # calculate total variation regularization (anisotropic version)
            # https://www.wikiwand.com/en/Total_variation_denoising
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT*(diff_i + diff_j)

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            agg_content_loss += content_loss.item()
            agg_style_loss += style_loss.item()
            agg_tv_loss += tv_loss.item()

            if (batch_num + 1) % LOG_INTERVAL == 0:
                iteration_count += LOG_INTERVAL
                iterations.append(iteration_count)
                avg_content_losses.append(agg_content_loss / LOG_INTERVAL)
                avg_style_losses.append(agg_style_loss / LOG_INTERVAL)
                avg_tv_losses.append(agg_tv_loss / LOG_INTERVAL)
                avg_total_losses.append((agg_content_loss + agg_style_loss + agg_tv_loss) / LOG_INTERVAL)

                status = "{}  Epoch {}:  [{}/{}]  Batch:[{}]  agg_style: {:.6f}  agg_content: {:.6f}  agg_tv: {:.6f}  style: {:.6f}  content: {:.6f}  tv: {:.6f} ".format(
                                time.ctime(), e + 1, img_count, len(train_dataset), batch_num+1,
                                agg_style_loss/(batch_num+1.0), agg_content_loss/(batch_num+1.0), agg_tv_loss/(batch_num+1.0),
                                style_loss.item(), content_loss.item(), tv_loss.item()
                            )
                print(status)

                agg_content_loss = agg_style_loss = agg_tv_loss = 0.0

    # Plot and save the graph
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, avg_content_losses, label='Content Loss')
    plt.plot(iterations, avg_style_losses, label='Style Loss')
    plt.plot(iterations, avg_tv_losses, label='TV Loss')
    plt.plot(iterations, avg_total_losses, label='Total Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss per iteration')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.close()

    # save model
    image_transformer.eval()

    image_transformer.to(device)

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = "models/" + str(style_name) + "_" + str(time.ctime()).replace(' ', '_') + ".model"
    torch.save(image_transformer.state_dict(), filename)

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
            # transforms.CenterCrop(512),             # crop center image_size out
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


def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size

    return width, height


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
