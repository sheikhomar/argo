#!/usr/bin/env python

"""
Visualisation of the filters.
Inspiration from: https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py
"""
import dl_models.lenet_01 as module
import data_helper as dataset
import utils as utils
import numpy as np
import argparse as ap

from os import path
from PIL import Image
from keras import backend as K
from keras.layers import Conv2D
from keras.preprocessing.image import save_img


def decode_image(x, image_shape):
    """
    Converts a tensor image to a valid image.
    :param x:
    :param image_shape:
    :return:
    """
    # normalize tensor: center on 0.,
    x -= x.mean()

    # Ensure std is 0.1
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to image array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    x = x.reshape(*image_shape)
    return x


def get_image_shape(input_tensor):
    """
    Extracts image shape from the input tensor
    :param input_tensor:
    :return:
    """
    width = input_tensor.shape[1].value
    height = input_tensor.shape[2].value
    channels = input_tensor.shape[3].value
    return width, height, channels


def generate_filter_image(input_tensor, output_tensor, filter_index, epochs=40, step=0.5):
    """Attempt to generate the input image that maximises the activation for one particular filter.

    :param input_tensor: The input image tensor
    :param output_tensor: The output image tensor
    :param filter_index: The filter to generate the image image for
    :param epochs: Number of iterations to run the gradient ascent
    :param step: Step size at each iteration
    :return:
    """

    # Set up an objective function that maximises the activation of a given filter
    loss = K.mean(output_tensor[:, :, :, filter_index])

    # Compute the gradient of the objective function w.r.t the input image tensor
    grads = K.gradients(loss, input_tensor)[0]

    # Normalise the gradient of the pixels of the input image to avoids very small and very large gradients.
    # This ensures that the gradient ascent process is smooth.
    grads = grads / (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

    # Define a function that computes the loss and the gradient given an input image
    iterate = K.function([input_tensor], [loss, grads])

    # Generate a gray image with some random noise
    img_shape = get_image_shape(input_tensor)
    img_data = np.random.random((1, *img_shape)) * 20 + 128.

    # Gradient ascent
    for _ in range(epochs):
        loss_value, grads_value = iterate([img_data])

        # Maximise the objective function so as to activate the filter as much as possible.
        # This allows us to visualise the patterns that the filter is detecting.
        img_data += grads_value * step

        # Quit if we get stuck
        if loss_value <= K.epsilon():
            break

    img = decode_image(img_data[0], img_shape)
    return img, loss_value


def save_filter_grid(file_path, filters, image_shape, margin=2):
    n = int(np.ceil(np.sqrt(len(filters))))

    filter_width = image_shape[0]
    filter_height = image_shape[1]
    channels = image_shape[2]

    width = n * filter_width + (n - 1) * margin
    height = n * filter_height + (n - 1) * margin
    stitched_filters = np.ones((width, height, channels), dtype='uint8') * 255

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            filter_index = i * n + j
            if filter_index >= len(filters):
                break

            filter_img = filters[filter_index]
            width_margin = (filter_width + margin) * i
            height_margin = (filter_height + margin) * j
            stitched_filters[
                width_margin: width_margin + filter_width,
                height_margin: height_margin + filter_height,
                :
            ] = filter_img

    # save the result to disk
    save_img(file_path, stitched_filters)


def parse_args():
    parser = ap.ArgumentParser(
        description='Visualises conv layers in a deep learning model.')
    parser.add_argument('--model', '-m',
                        type=utils.verify_and_import_model,
                        required=True,
                        help='Supported models: %s' % ', '.join(utils.MODELS))
    return parser.parse_args()


def visualise_conv_layer(model, model_name, layer_index):
    layer = model.layers[layer_index]
    assert isinstance(layer, Conv2D)

    layer_name = layer.name
    print(f'Visualising layer {layer_name}!')

    input_tensor = model.inputs[0]
    output_tensor = layer.output
    image_shape = get_image_shape(input_tensor)

    processed_filters = []
    for filter_index in range(layer.filters):

        img, loss_val = generate_filter_image(input_tensor, output_tensor, filter_index)
        processed_filters.append(img)

    grid_path = f'./figures/{model_name}_{layer_name}_filters.png'
    print(f'Saving visualisation {grid_path}')
    save_filter_grid(grid_path, processed_filters, image_shape)


def visualise_model(model, model_name):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            visualise_conv_layer(model, model_name, i)


def main():
    args = parse_args()
    model_module = args.model
    model_name = model_module.MODEL_NAME
    print(f'Using model "{model_name}"...')

    model = model_module.get_model(dataset)
    weights_path = utils.get_weights_path(model_name)

    if not path.exists(weights_path):
        print(f'Could not find the weights file: {weights_path}')
        return

    print(f'Loading weights from {weights_path}...')
    model.load_weights(weights_path)

    visualise_model(model, model_name)


if __name__ == '__main__':
    main()
