from keras.datasets import mnist
import numpy as np
import skimage.morphology
import os
import cv2
import skimage.io
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage import measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from keras import Sequential
from keras.models import load_model
from PIL import Image
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D, BatchNormalization, Dropout
#from keras.utils import to_catagorical
import keras.utils
from keras.optimizers import Adam
from numpy import *
from scipy.ndimage.interpolation import shift
from utilities import *

def shift_image(image, dx, dy):
    image = np.asarray(image.copy()).reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image

if __name__ == '__main__':
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

    # remove digit 9 in training and testing set
    idx_9 = y_train_mnist == 9
    x_train = x_train_mnist[np.logical_not(idx_9),:,:]
    y_train = y_train_mnist[np.logical_not(idx_9)]
    idx_9 = y_test_mnist == 9
    x_test = x_test_mnist[np.logical_not(idx_9), :, :]
    y_test = y_test_mnist[np.logical_not(idx_9)]

    x_train_aug,y_train_aug = x_train.tolist(), y_train.tolist()
    # data augmentation for training
    idx_aug = 0
    for image,label in zip(x_train,y_train):
        # shift
        dx = (np.random.rand() < 0.5)*(-3) + (np.random.rand() > 0.5)*3
        dy = (np.random.rand() < 0.5) * (-3) + (np.random.rand() > 0.5) * 3
        idx_aug += 1
        print('Augmented {} images...'.format(idx_aug))
        t_shift = shift_image(image, dx, dy) # uint8
        x_train_aug.append(t_shift)
        y_train_aug.append(label)
        # rotate
        for angle in range(-180,210,30):
            if angle != 0:
                idx_aug += 1
                print('Augmented {} images...'.format(idx_aug))
                # x_train_aug.append(rotate(t_shift,angle))
                t_rot = skimage.transform.rotate(t_shift,angle,preserve_range=True).astype('uint8')
                t_blur = skimage.filters.gaussian(t_rot, sigma=np.random.rand() + 0.5, preserve_range=True).astype('uint8')
                x_train_aug.append(t_blur)
                y_train_aug.append(label)

    x_train_aug = np.asarray(x_train_aug)
    y_train_aug = np.asarray(y_train_aug)

    x_train,y_train = x_train_aug,y_train_aug

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    y_test = keras.utils.to_categorical(y_test, 9)
    y_train = keras.utils.to_categorical(y_train, 9)

    # design model
    model = Sequential()
    model.add(Convolution2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Convolution2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(9, activation='softmax'))
    adam = Adam(lr=0.001)
    # compile model
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # training model
    model.fit(x_train, y_train, batch_size=100, epochs=10)
    # save model
    model.save('./nn_digit.h5')
