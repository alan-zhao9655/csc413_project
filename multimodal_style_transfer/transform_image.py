# Imports
import sys
import time
import os
import torch
from utils import *

def main():
    # Style model
    MODEL = "picasso"

    # Load input image
    CONTENT = "dan"
    input_img_path = "content_imgs/" + CONTENT + ".jpg"
    input_img = image_loader(input_img_path)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    style_subnet = torch.load('models/trained_style_subnet' + "_" + MODEL + '.pt', map_location='cpu').eval().to(device)
    enhance_subnet = torch.load('models/trained_enhance_subnet' + "_" + MODEL + '.pt', map_location='cpu').eval().to(device)
    refine_subnet = torch.load('models/trained_refine_subnet' + "_" + MODEL + '.pt', map_location='cpu').eval().to(device)

    # Transform
    print("Start transforming on {}..".format(device))
    start = time.time()
    with torch.no_grad():
        generated_img_256, resized_input_img_256 = style_subnet(input_img)
        generated_img_512, resized_input_img_512 = enhance_subnet(generated_img_256)
        generated_img_1024, resized_input_img_1024 = refine_subnet(generated_img_512)
    print("Image transformed. Time for pass: {:.2f}s".format(time.time() - start))

    # save_image(generated_img_256, title="generated/multimodal_" + MODEL + '_' + CONTENT + "_256")
    # save_image(generated_img_512, title="generated/multimodal_" + MODEL + '_' + CONTENT + "_512")
    save_image(generated_img_1024, title="generated/multimodal_" + MODEL + '_' + CONTENT + "_1024")

if __name__ == "__main__":
    main()
