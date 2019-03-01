#!/usr/bin/env python

import argparse as ap
import data_helper as dataset
import utils as utils

from keras.callbacks import TensorBoard


def parse_args():
    parser = ap.ArgumentParser(
        description='Trains a deep learning model.')
    parser.add_argument('--model', '-m',
                        type=utils.verify_and_import_model,
                        required=True,
                        help='Supported models: %s' % ', '.join(utils.MODELS))
    return parser.parse_args()


def save_architecture(model_name, model):
    model_json = model.to_json()
    with open(f'./{utils.MODELS_DIR}/{model_name}.json', 'w') as json_file:
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

    model.save_weights(utils.get_weights_path(model_name), overwrite=False)


if __name__ == '__main__':
    main()
