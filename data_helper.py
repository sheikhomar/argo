from keras.datasets import mnist
import keras


NUM_CLASSES = 10
IMAGE_SHAPE = 28, 28, 1


def load():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, *IMAGE_SHAPE).astype('float32')
    X_test = X_test.reshape(10000, *IMAGE_SHAPE).astype('float32')

    X_train /= 255
    X_test /= 255

    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

    return X_train, y_train, X_test, y_test
