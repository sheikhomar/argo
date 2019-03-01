from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten, MaxPooling2D, Conv2D
from os import path

MODEL_NAME = path.splitext(path.basename(__file__))[0]


def get_model(dataset):
    model = Sequential(name='LeNet-01')
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=dataset.IMAGE_SHAPE))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(dataset.NUM_CLASSES, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
