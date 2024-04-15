import os
import argparse
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
    parser = argparse.ArgumentParser(description="Evaluate the Style Transfer Model")
    parser.add_argument("--content", type=str, required=True, help="Path to the content image")
    parser.add_argument("--style", type=str, required=True, help="Path to the style image")
    parser.add_argument("--generated", type=str, required=True, help="Path to the generated image")
    args = parser.parse_args()

    model = get_model()
    content_image = load_and_process_img(args.content)
    style_image = load_and_process_img(args.style)
    generated_image = load_and_process_img(args.generated)

    content_features = get_features(content_image, model, [CONTENT_LAYER])[CONTENT_LAYER]
    style_features = get_features(style_image, model, STYLE_LAYERS)
    gen_features = get_features(generated_image, model, STYLE_LAYERS + [CONTENT_LAYER])

    print("Content Loss: {:.2f}".format(tf.reduce_mean(tf.square(gen_features[CONTENT_LAYER] - content_features)).numpy()))
    style_loss = 0
    for layer in STYLE_LAYERS:
        style_loss += tf.reduce_mean(tf.square(gen_features[layer] - style_features[layer])).numpy()
    print("Style Loss: {:.2f}".format(style_loss))

if __name__ == "__main__":
    main()
