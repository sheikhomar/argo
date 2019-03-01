import os
import importlib
from os import path
import argparse as ap

MODELS_DIR = 'dl_models'
MODELS = [path.splitext(f)[0]
          for f in os.listdir(MODELS_DIR)
          if path.isfile(path.join(MODELS_DIR, f))]


def get_weights_path(model_name):
    return f'./dl_weights/{model_name}.hdf5'


def verify_and_import_model(model_name):
    if model_name not in MODELS:
        msg = 'Unknown algorithm "%s"!' % model_name
        raise ap.ArgumentTypeError(msg)
    return importlib.import_module(f'{MODELS_DIR}.{model_name}')
