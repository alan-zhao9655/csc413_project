import os
import cv2

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
    models = {
        "perceptualLoss_style_transfer": {"base_path": "perceptualLoss_style_transfer"},
        "multimodal_style_transfer": {"base_path": "multimodal_style_transfer"},
        "CNN": {"base_path": "CNN"}
    }

    content_images = ["dan.jpg", "toronto.jpg", "maja.jpg"]
    style_images = ["forest1024.jpg", "picasso.jpg", "starry_night.jpg", "mosaic.jpg"]

    # Open a text file to write the losses
    with open("image_quality_evaluations.txt", "w") as file:
        for model_name, info in models.items():
            base_path = info["base_path"]
            file.write(f"Model: {model_name}\n")

            for content_image in content_images:
                content_path = os.path.join(base_path, "content_imgs", content_image)
                content_image_score = calculate_brisque_score(content_path)
                file.write(f"\nContent Image: {content_image}, BRISQUE Score: {content_image_score}\n")

                for style_image in style_images:
                    style_path = os.path.join(base_path, "style_imgs", style_image)
                    style_image_score = calculate_brisque_score(style_path)

                    generated_image_name = f"{os.path.splitext(style_image)[0]}_{os.path.splitext(content_image)[0]}.jpg"
                    if model_name == "multimodal_style_transfer":
                        generated_image_name = f"multimodal_{os.path.splitext(style_image)[0]}_{os.path.splitext(content_image)[0]}_1024.jpg"
                    generated_path = os.path.join(base_path, "output", generated_image_name)
                    generated_image_score = calculate_brisque_score(generated_path)

                    ssim_score = calculate_ssim_score(content_path, generated_path)
                    psnr_score = calculate_psnr_score(content_path, generated_path)
                    gmsd_score = calculate_gmsd_score(content_path, generated_path)

                    file.write(f"  Style Image: {style_image}, Generated Image: {generated_image_name}\n")
                    file.write(f"  Style Image BRISQUE Score: {style_image_score}\n")
                    file.write(f"  Generated Image BRISQUE Score: {generated_image_score}\n")
                    file.write(f"  SSIM: {ssim_score}, PSNR: {psnr_score}, GMSD: {gmsd_score}\n")

if __name__ == "__main__":
    main()