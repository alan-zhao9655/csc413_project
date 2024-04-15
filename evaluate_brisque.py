import os
import cv2
from skimage import io
from sewar.full_ref import mse, ssim
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
from brisque import BRISQUE
import argparse

def calculate_brisque_score(image_path):
    # https://github.com/opencv/opencv_contrib/tree/master/modules/quality
    img = cv2.imread(image_path)
    model_path = '/usr/share/opencv4/quality/brisque_model_live.yml'
    range_path = '/usr/share/opencv4/quality/brisque_range_live.yml'
    # if len(image.shape) == 2:  # Image is grayscale
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    obj = cv2.quality.QualityBRISQUE_create(model_path, range_path)
    score = obj.compute(img)
    return score[0]

def load_and_resize(image_path, width=None, height=None):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image didn't load correctly. Check the image path.")
    if width is not None and height is not None:
        img = cv2.resize(img, (width, height))  # Resize image to match dimensions
    return img

def calculate_ssim_score(content_img_path, generated_img_path):
    content_img = load_and_resize(content_img_path)
    generated_img = load_and_resize(generated_img_path, content_img.shape[1], content_img.shape[0])

    ssim_scalar, map = cv2.quality.QualitySSIM_compute(content_img, generated_img)
    return ssim_scalar[0]

def calculate_psnr_score(content_img_path, generated_img_path):
    content_img = load_and_resize(content_img_path)
    generated_img = load_and_resize(generated_img_path, content_img.shape[1], content_img.shape[0])

    psnr_value = cv2.quality.QualityPSNR_compute(content_img, generated_img)
    return psnr_value[0]

def calculate_gmsd_score(content_img_path, generated_img_path):
    content_img = load_and_resize(content_img_path)
    generated_img = load_and_resize(generated_img_path, content_img.shape[1], content_img.shape[0])

    gmsd_value = cv2.quality.QualityGMSD_compute(content_img, generated_img)
    return gmsd_value[0]

def main():
    parser = argparse.ArgumentParser(description="Image Quality Evaluation")
    parser.add_argument("--content", required=True, help="Path to content image")
    parser.add_argument("--style", required=True, help="Path to style image")
    parser.add_argument("--generated", required=True, help="Path to generated (styled) image")
    args = parser.parse_args()

    content_image_score = calculate_brisque_score(args.content)
    style_image_score = calculate_brisque_score(args.style)
    generated_image_score = calculate_brisque_score(args.generated)

    print("Evaluating Image Quality...")
    print(f"Content Image BRISQUE Score: {content_image_score}")
    print(f"Style Image BRISQUE Score: {style_image_score}")
    print(f"Generated Image BRISQUE Score: {generated_image_score}")

    # Compare content and generated images for reference
    ssim_score = calculate_ssim_score(args.content, args.generated)
    psnr_score = calculate_psnr_score(args.content, args.generated)
    gmsd_score = calculate_gmsd_score(args.content, args.generated)
    print(f"PSNR between Content and Generated: {psnr_score}")
    print(f"SSIM between Content and Generated: {ssim_score}")
    print(f"GMSD between Content and Generated: {gmsd_score}")

if __name__ == "__main__":
    main()