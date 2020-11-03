import numpy as np
import os
import cv2
import skimage.transform
import skimage.filters
from keras import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D, BatchNormalization, Dropout
from scipy.ndimage.interpolation import shift

from utilities import *

def shift_image(image, dx, dy):
    image = np.asarray(image.copy()).reshape((32, 32, 3))
    shifted_image = shift(image, [dy, dx,0], cval=255, mode="constant")
    return shifted_image

if __name__ == '__main__':
    operators_path = './data/operators/training_set'
    operators_names = [nm for nm in os.listdir(operators_path) if '.png' in nm]  # make sure to only load .png
    operators_names.sort()  # sort file names

    operators_im, labels_im = [],[]
    for nm in operators_names:
        sm = cv2.imread(os.path.join(operators_path, nm))
        # sm = cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY)
        sm = cv2.resize(sm, (32, 32), interpolation=cv2.INTER_AREA)
        operators_im.append(sm)
        if 'division' in nm:
            labels_im.append(0)
        elif 'equal' in nm:
            labels_im.append(1)
        elif 'minus' in nm:
            labels_im.append(2)
        elif 'multiplication' in nm:
            labels_im.append(3)
        elif 'plus' in nm:
            labels_im.append(4)

    # data augmentation
    operators_aug, labels_aug = [],[]
    # operators_div, operators_eql, operators_minu, operators_mul, operators_add = [], [], [], [], []
    print('Start to augment the dataset...')
    for j in range(len(operators_im)):
        operators_aug.append([])
        labels_aug.append([])
        cur_label = labels_im[j]
        for i in range(360):
            t_rot = rotate(operators_im[j], i)
            t_rot = cv2.resize(t_rot, (32, 32), interpolation=cv2.INTER_AREA)
            t_rot_gray = cv2.cvtColor(t_rot,cv2.COLOR_BGR2GRAY)
            operators_aug[j].append(t_rot_gray) # uint8
            labels_aug[j].append(cur_label)

            for ii in range(3):
                dx = np.random.rand() * 4 - 2
                dy = np.random.rand() * 4 - 2
                t_shift = shift_image(t_rot, dx, dy) # uint8
                t_shift_gray = cv2.cvtColor(t_shift, cv2.COLOR_BGR2GRAY)
                operators_aug[j].append(t_shift_gray)
                labels_aug[j].append(cur_label)
                for jj in range(3):
                    t_blur = skimage.filters.gaussian(t_shift, sigma=np.random.rand()*0.5 + 0.1, preserve_range=True).astype('uint8')
                    t_gamma = skimage.exposure.adjust_gamma(t_blur, gamma=np.random.rand()+0.5, gain=np.random.rand()+0.5).astype('uint8')
                    t_gamma_gray = cv2.cvtColor(t_gamma, cv2.COLOR_BGR2GRAY)
                    operators_aug[j].append(t_gamma_gray) # uint8
                    labels_aug[j].append(cur_label)

    for j in range(len(operators_aug)):
        operators_aug[j] = np.asarray(operators_aug[j])
        labels_aug[j] = np.asarray(labels_aug[j])

    ope = np.concatenate(operators_aug,axis=0)
    label_ope = np.concatenate(labels_aug,axis=0)
    label_ope = keras.utils.to_categorical(label_ope, 5)
    ope = ope.reshape((len(ope), 32, 32, 1))
    x_train = ope
    y_train = label_ope

    # design model
    # model = Sequential()
    # model.add(Convolution2D(8, (3, 3), input_shape=(32, 32, 1)))
    # model.add(MaxPooling2D(2, 2))
    # model.add(Activation('relu'))
    # model.add(Flatten())
    #
    # model.add(Dense(360))
    # model.add(Activation('relu'))
    # model.add(Dense(5))
    # model.add(Activation('softmax'))
    # adam = Adam(lr=0.0001)

    model = Sequential()
    model.add(Convolution2D(32, kernel_size=3, activation='relu', input_shape=(32, 32, 1)))
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
    model.add(Dense(5, activation='softmax'))
    adam = Adam(lr=0.001)


    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    print('Start to train the operator classifier...')
    model.fit(x_train, y_train, batch_size=100, epochs=5)
    model.save('./nn_operator.h5')
    print('Training finished!')

