#!/usr/bin/env python

import os
import importlib
import argparse as ap
import data_helper as dataset

from os import path
from keras.callbacks import TensorBoard


MODELS_DIR = 'dl_models'
MODELS = [path.splitext(f)[0]
          for f in os.listdir(MODELS_DIR)
          if path.isfile(path.join(MODELS_DIR, f))]


def import_module(model_name):
    if model_name not in MODELS:
        msg = 'Unknown algorithm "%s"!' % model_name
        raise ap.ArgumentTypeError(msg)
    return importlib.import_module(f'{MODELS_DIR}.{model_name}')


def parse_args():
    parser = ap.ArgumentParser(
        description='Trains a deep learning model.')
    parser.add_argument('--model', '-m',
                        type=import_module,
                        required=True,
                        help='Supported models: %s' % ', '.join(MODELS))
    return parser.parse_args()


def save_architecture(model_name, model):
    model_json = model.to_json()
    with open(f'./{MODELS_DIR}/{model_name}.json', 'w') as json_file:
        json_file.write(model_json)


def main():
    args = parse_args()
    module = args.model
    model_name = module.MODEL_NAME
    print(f'Using model "{model_name}"...')

    model = module.get_model(dataset)
    model.summary()

    print('Loading data...')
    X_train, y_train, X_test, y_test = dataset.load()

    print('Training...')
    tb = TensorBoard(f'./logs/{model_name}')
    model.fit(X_train, y_train, batch_size=128, epochs=15, verbose=1,
              validation_data=(X_test, y_test), callbacks=[tb])

    print('Saving weights')
    model.save_weights(f'./dl_weights/${model_name}.hdf5', overwrite=False)


if __name__ == '__main__':
    main()
