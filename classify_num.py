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
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D
#from keras.utils import to_catagorical
import keras.utils
from keras.optimizers import Adam
from numpy import *

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train.astype("float32")/255
x_test=x_test.astype("float32")/255
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

y_test =keras.utils.to_categorical(y_test, 10)
y_train = keras.utils.to_categorical(y_train, 10)

# design model
model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(64*28))
model.add(Activation('relu'))
model.add(Dense(240))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
adam = Adam(lr=0.001)
# compile model
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# training model
model.fit(x_train, y_train, batch_size=100, epochs=5)
# test model
#print(model.evaluate(x_test, y_test, batch_size=100))
# save model
# model.save('/Users/maxiaoqi/Desktop/iapr_2020_master/my_model2.h5')
model.save('./my_model2.h5')
