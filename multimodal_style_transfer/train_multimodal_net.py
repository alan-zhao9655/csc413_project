from collections import namedtuple
import time
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from torch.autograd import Variable
from torchvision.models import vgg19, VGG19_Weights

from utils import *
from style_subnet import *
from enhance_subnet import *
from refine_subnet import *
from PIL import Image, ImageFile
from simple_dataset import SimpleDataset

""" Allow PIL to read truncated blocks when loading images """
ImageFile.LOAD_TRUNCATED_IMAGES = True

""" Add a seed to have reproducable results """
SEED = 1080
torch.manual_seed(SEED)

def train():
    IMAGE_SIZE = 256
    BATCH_SIZE = 1
    STYLE_NAME = "picasso"
    LR = 1e-3
    NUM_EPOCHS = 1
    CONTENT_WEIGHTS = [1, 1, 1]
    STYLE_WEIGHTS = [2e4, 1e5, 1e3] # Checkpoint single style
    #STYLE_WEIGHTS = [5e4, 8e4, 3e4] # Checkpoint two styles
    LAMBDAS = [1., 0.5, 0.25]
    REG = 1e-7
    LOG_INTERVAL = 100

    """ Configure training with or without cuda """

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed(SEED)
        torch.set_default_device(device)
    else:
        device = torch.device("cpu")
        torch.set_default_device(device)

    """ Load coco dataset """
    script_dir = os.path.dirname(os.path.realpath('__file__'))
    DATASET_PATH = script_dir + "/../train2014-2"
    dataset_transform = transforms.Compose([transforms.Resize(IMAGE_SIZE),
                                    transforms.CenterCrop(IMAGE_SIZE),
                                    transforms.ToTensor(), tensor_normalizer()])
    # Initialize dataset
    train_dataset = SimpleDataset(DATASET_PATH, dataset_transform)
    print(len(train_dataset))

    # Initialize DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    """ Load Style Image """
    style_img_256, style_img_512, style_img_1024 = style_loader(
        "style_imgs/" + STYLE_NAME + ".jpg", device, [256, 512, 1024])

    """ Define Loss Network """
    StyleOutput = namedtuple("StyleOutput", ["relu1_1", "relu2_1", "relu3_1", "relu4_1"])
    ContentOutput = namedtuple("ContentOutput", ["relu2_1"])

    # https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
    class LossNetwork(torch.nn.Module):
        def __init__(self, vgg):
            super(LossNetwork, self).__init__()
            self.vgg = vgg
            self.layer_name_mapping = {
                '1': "relu1_1", '3': "relu1_2",
                '6': "relu2_1", '8': "relu2_2",
                '11': "relu3_1", '13': "relu3_2", '15': "relu3_3", '17': "relu3_4",
                '20': "relu4_1", '22': "relu4_2", '24': "relu4_3", '26': "relu4_4",
                '29': "relu5_1", '31': "relu5_2", '33': "relu5_3", '35': "relu5_4"
            }

        def forward(self, x, mode):
            if mode == 'style':
                layers = ['1', '6', '11', '20']
            elif mode == 'content':
                layers = ['6']
            else:
                print("Invalid mode. Select between 'style' and 'content'")
            output = {}
            for name, module in self.vgg._modules.items():
                x = module(x)
                if name in layers:
                    output[self.layer_name_mapping[name]] = x
            if mode == 'style':
                return StyleOutput(**output)
            else:
                return ContentOutput(**output)

    """ Load and extract features from VGG16 """
    print("Loading VGG..")
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    loss_network = LossNetwork(vgg).to(device).eval()
    del vgg

    """ Before training, compute the features of every resolution of the style image """
    print("Computing style features..")
    with torch.no_grad(): 
        style_loss_features_256 = loss_network(Variable(style_img_256), 'style')
        style_loss_features_512 = loss_network(Variable(style_img_512), 'style')
        style_loss_features_1024 = loss_network(Variable(style_img_1024), 'style')
    gram_style_256 = [Variable(gram_matrix(y).data, requires_grad=False) for y in style_loss_features_256]
    gram_style_512 = [Variable(gram_matrix(y).data, requires_grad=False) for y in style_loss_features_512]
    gram_style_1024 = [Variable(gram_matrix(y).data, requires_grad=False) for y in style_loss_features_1024]

    """ Init Net and loss """
    style_subnet = StyleSubnet().to(device)
    enhance_subnet = EnhanceSubnet().to(device)
    refine_subnet = RefineSubnet().to(device)

    """ Prepare Training """
    max_iterations = min(10000, len(train_dataset))

    # init loss
    mse_loss = torch.nn.MSELoss()
    # init optimizer
    optimizer = torch.optim.Adam(list(style_subnet.parameters()) + 
                                list(enhance_subnet.parameters()) +
                                list(refine_subnet.parameters()), lr=LR)

    def getLosses(generated_img, resized_input_img, content_weight, style_weight, mse_loss, gram_style):
        
        # Compute features
        generated_style_features = loss_network(generated_img, 'style')
        generated_content_features = loss_network(generated_img, 'content')
        target_content_features = loss_network(resized_input_img, 'content')
        
        # Content loss
        target_content_features = Variable(target_content_features[0].data, requires_grad=False)
        content_loss = content_weight * mse_loss(generated_content_features[0], target_content_features)
        
        # Style loss
        style_loss = 0.
        for m in range(len(generated_style_features)):
            gram_s = gram_style[m]
            gram_y = gram_matrix(generated_style_features[m])
            style_loss += style_weight * mse_loss(gram_y, gram_s.expand_as(gram_y))
        
        # Regularization loss
        reg_loss = REG * (
            torch.sum(torch.abs(generated_img[:, :, :, :-1] - generated_img[:, :, :, 1:])) + 
            torch.sum(torch.abs(generated_img[:, :, :-1, :] - generated_img[:, :, 1:, :])))
        
        return content_loss, style_loss, reg_loss

    """ Perform Training """
    style_subnet.train()
    enhance_subnet.train()
    refine_subnet.train()
    start = time.time()
    print("Start training on {}...".format(device))

    # Initialize lists to store average losses for plotting
    avg_content_losses, avg_style_losses, avg_reg_losses, avg_total_losses = [], [], [], []
    for epoch in range(NUM_EPOCHS):
        agg_content_loss, agg_style_loss, agg_reg_loss = 0., 0., 0.
        log_counter = 0
        for i, x in enumerate(train_loader):
            # update learning rate every 2000 iterations
            if i % 2000 == 0 and i != 0:
                LR = LR * 0.8
                optimizer = torch.optim.Adam(list(style_subnet.parameters()) + 
                                            list(enhance_subnet.parameters()) +
                                            list(refine_subnet.parameters()), lr=LR)
            
            
            optimizer.zero_grad()
            x_in = x.clone()
            
            """ Style Subnet """
            x_in = Variable(x_in).to(device)

            # Generate image
            generated_img_256, resized_input_img_256 = style_subnet(x_in)
            resized_input_img_256 = Variable(resized_input_img_256.data)
            
            # Compute Losses
            style_subnet_content_loss, style_subnet_style_loss, style_subnet_reg_loss = getLosses(
                generated_img_256,
                resized_input_img_256,
                CONTENT_WEIGHTS[0],
                STYLE_WEIGHTS[0],
                mse_loss, gram_style_256)
            
            """ Enhance Subnet """
            x_in = Variable(generated_img_256)
            
            # Generate image
            generated_img_512, resized_input_img_512 = enhance_subnet(x_in)
            resized_input_img_512 = Variable(resized_input_img_512.data)
            
            # Compute Losses
            enhance_subnet_content_loss, enhance_subnet_style_loss, enhance_subnet_reg_loss = getLosses(
                generated_img_512,
                resized_input_img_512,
                CONTENT_WEIGHTS[1],
                STYLE_WEIGHTS[1],
                mse_loss, gram_style_512)

            """ Refine Subnet """
            x_in = Variable(generated_img_512)
            
            # Generate image
            generated_img_1024, resized_input_img_1024 = refine_subnet(x_in)
            resized_input_img_1024 = Variable(resized_input_img_1024.data)
            
            # Compute Losses
            refine_subnet_content_loss, refine_subnet_style_loss, refine_subnet_reg_loss = getLosses(
                generated_img_1024,
                resized_input_img_1024,
                CONTENT_WEIGHTS[2],
                STYLE_WEIGHTS[2],
                mse_loss, gram_style_1024)
            
            # Total loss
            total_loss = LAMBDAS[0] * (style_subnet_content_loss + style_subnet_style_loss + style_subnet_reg_loss) + \
                        LAMBDAS[1] * (enhance_subnet_content_loss + enhance_subnet_style_loss + enhance_subnet_reg_loss) + \
                        LAMBDAS[2] * (refine_subnet_content_loss + refine_subnet_style_loss + refine_subnet_reg_loss)
            total_loss.backward()
            optimizer.step()

            # Aggregated loss
            agg_content_loss += style_subnet_content_loss.item() + \
                                enhance_subnet_content_loss.item() + \
                                refine_subnet_content_loss.item()
            agg_style_loss += style_subnet_style_loss.item() + \
                            enhance_subnet_style_loss.item() + \
                            refine_subnet_style_loss.item()
            
            agg_reg_loss += style_subnet_reg_loss.item() + \
                            enhance_subnet_reg_loss.item() + \
                            refine_subnet_reg_loss.item()
            
            # log training process
            if (i + 1) % LOG_INTERVAL == 0:
                log_counter += 1
                hlp = log_counter * LOG_INTERVAL
                time_per_pass = (time.time() - start) / hlp
                estimated_time_left = (time_per_pass * (max_iterations - i))/3600

                avg_content_loss = agg_content_loss / LOG_INTERVAL
                avg_style_loss = agg_style_loss / LOG_INTERVAL
                avg_reg_loss = agg_reg_loss / LOG_INTERVAL
                avg_total_loss = (agg_content_loss + agg_style_loss + agg_reg_loss) / LOG_INTERVAL

                print("{} [{}/{}] time per pass: {:.2f}s  total time: {:.2f}s  estimated time left: {:.2f}h  content: {:.6f}  style: {:.6f}  reg: {:.6f}  total: {:.6f}".format(
                            time.ctime(), i+1, max_iterations,
                            (time.time() - start) / (i + 1),
                            time.time() - start,
                            estimated_time_left,
                            avg_content_loss,
                            avg_style_loss,
                            avg_reg_loss,
                            avg_total_loss))
                
                # Append average losses for plotting
                avg_content_losses.append(avg_content_loss)
                avg_style_losses.append(avg_style_loss)
                avg_reg_losses.append(avg_reg_loss)
                avg_total_losses.append(avg_total_loss)
            

                agg_content_loss, agg_style_loss, agg_reg_loss = 0., 0., 0.

            # Stop training after max iterations
            if (i + 1) == max_iterations: break

    # Plotting the average loss graph at the end of an epoch
    plt.figure(figsize=(10, 5))
    plt.plot(avg_content_losses, label='Average Content Loss')
    plt.plot(avg_style_losses, label='Average Style Loss')
    plt.plot(avg_reg_losses, label='Average Regularization Loss')
    plt.plot(avg_total_losses, label='Average Total Loss')
    plt.xlabel('Number of Intervals')
    plt.ylabel('Average Loss Value')
    plt.title('Average Training Losses Over Intervals')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    plt.close()     

    """ Save model """
    torch.save(style_subnet, 'models/trained_style_subnet_' + STYLE_NAME + '.pt')
    torch.save(enhance_subnet, 'models/trained_enhance_subnet_' + STYLE_NAME + '.pt')
    torch.save(refine_subnet, 'models/trained_refine_subnet_' + STYLE_NAME + '.pt')

if __name__ == "__main__":
    train()
