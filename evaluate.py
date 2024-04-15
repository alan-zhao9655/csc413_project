import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
CONTENT_LAYER = 'block5_conv2'
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

def load_and_process_img(img_path):
    img = load_img(img_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def get_features(image, model, layers):
    outputs = model(image)
    return {layer_name: outputs[layer_name] for layer_name in layers}

def get_model():
    model = vgg19.VGG19(weights='imagenet', include_top=False)
    outputs = {layer.name: layer.output for layer in model.layers if layer.name in STYLE_LAYERS + [CONTENT_LAYER]}
    return tf.keras.Model(inputs=model.inputs, outputs=outputs)

def main():
    import os
    MODEL = "multimodal_style_transfer"
    content_images = [
        "content_imgs/dan.jpg",
        "content_imgs/toronto.jpg",
        "content_imgs/maja.jpg"
    ]
    style_images = [
        "style_imgs/forest1024.jpg",
        "style_imgs/picasso.jpg",
        "style_imgs/starry_night.jpg",
        "style_imgs/mosaic.jpg"
    ]

    model = get_model()
    loss_filename = MODEL + '_' + 'losses.txt'
    with open(loss_filename, "w") as file:
        # Iterate over all content and style images
        for content_filename in content_images:
            content_path = os.path.join(MODEL, content_filename)
            content_image = load_and_process_img(content_path)
            content_features = get_features(content_image, model, [CONTENT_LAYER])[CONTENT_LAYER]

            for style_filename in style_images:
                style_path = os.path.join(MODEL, style_filename)
                style_image = load_and_process_img(style_path)
                style_features = get_features(style_image, model, STYLE_LAYERS)

                # Generate a unique filename for each generated image
                base_content_name = os.path.splitext(os.path.basename(content_filename))[0]
                base_style_name = os.path.splitext(os.path.basename(style_filename))[0]
                # generated_filename = f"{base_style_name}_{base_content_name}.jpg"
                generated_filename = f"multimodal_{base_style_name}_{base_content_name}_1024.jpg" # for multimodal only
                generated_path = os.path.join(MODEL, "output", generated_filename)

                generated_image = load_and_process_img(generated_path)
                gen_features = get_features(generated_image, model, STYLE_LAYERS + [CONTENT_LAYER])

                content_loss = tf.reduce_mean(tf.square(gen_features[CONTENT_LAYER] - content_features)).numpy()
                style_loss = 0
                for layer in STYLE_LAYERS:
                    style_loss += tf.reduce_mean(tf.square(gen_features[layer] - style_features[layer])).numpy()

                # Write to file
                file.write(f"Content Image: {os.path.basename(content_path)}, Style Image: {os.path.basename(style_path)}, Generated Image: {generated_filename}\n")
                file.write(f"Content Loss: {content_loss:.2f}, Style Loss: {style_loss:.2f}\n\n")

if __name__ == "__main__":
    main()
