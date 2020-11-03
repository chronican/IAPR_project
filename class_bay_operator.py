import pylab
import imageio
import skimage
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D
#from keras.utils import to_catagorical
import keras.utils
from keras.optimizers import Adam
import numpy as np
from keras.models import load_model
import numpy as np
from PIL import Image
import numpy as np
import skimage.io
from sklearn.neural_network import MLPClassifier
from skimage.transform import resize
from skimage.transform import rotate
from skimage.util import pad
import time
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
import numpy as np
import skimage.morphology
import os
import skimage.io
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage import measure
from skimage.segmentation import watershed
from scipy import ndimage as ndi
import cv2
from numpy import *
from numpy.linalg import inv,det

operators_path = './data/operators'
operators_names = [nm for nm in os.listdir(operators_path) if '.png' in nm]  # make sure to only load .png
operators_names.sort()  # sort file names
# print(os.path.join(operators_path, operators_names[0]))

operators_cm = []
for nm in operators_names:
    sm=cv2.imread(os.path.join(operators_path, nm))
    sm=cv2.cvtColor(sm,cv2.COLOR_BGR2GRAY)
    sm=cv2.resize(sm, (32, 32), interpolation=cv2.INTER_AREA)
    sm=sm/255
    operators_cm.append(sm)
# print(operators_cm[1])


def rotate(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(1, 1, 1))
    return imgRotation

def watershed_edge(im):
    tmp = im.copy()
    tmp.astype(float)
    tmp = (tmp-tmp.min())/(tmp.max()-tmp.min())*255
    elevation_map = sobel(tmp)
    markers = np.zeros_like(tmp)
    markers[tmp<10] = 1
    markers[tmp>200] = 2
    tmp = watershed(elevation_map,markers)-1
    return tmp

def get_contour(im):
    tmp = skimage.measure.find_contours(im,0.5)
    x,y = np.round(tmp[0][:,1]),np.round(tmp[0][:,0])
    z = np.empty_like(x,dtype='complex')
    for i in range(len(x)):
        z[i] = np.complex(x[i],y[i])
    return x,y,z
def g(x,S,m):
    tmp =(-1/2) * (x-m).transpose().dot(inv(S)).dot(x-m)+np.log(1/(2*np.pi*np.linalg.det(S))**0.5)-np.log(1/(2*np.pi*det(S))**0.5)
    return tmp

operators_div=np.zeros((360,32,32))
operators_eql=np.zeros((360,32,32))
operators_minu=np.zeros((360,32,32))
operators_mul=np.zeros((360,32,32))
operators_add=np.zeros((360,32,32))

for i in range (360):
    a=rotate(operators_cm[0],i)
    operators_div[i]=cv2.resize(a,(32,32),interpolation=cv2.INTER_AREA)
    b=rotate(operators_cm[1],i)
    operators_eql[i]=cv2.resize(b,(32,32),interpolation=cv2.INTER_AREA)
    c=rotate(operators_cm[2],i)
    operators_minu[i]=cv2.resize(c,(32,32),interpolation=cv2.INTER_AREA)
    d=rotate(operators_cm[3],i)
    operators_mul[i]=cv2.resize(d,(32,32),interpolation=cv2.INTER_AREA)
    e=rotate(operators_cm[4],i)
    operators_add[i]=cv2.resize(e,(32,32),interpolation=cv2.INTER_AREA)

operators_fft_minu=np.ones([360,4],dtype='complex')
for im,ii in zip(operators_minu,range(360)):
    x,y,z = get_contour(im)
    fft= np.fft.fft(z)[0:4]
    operators_fft_minu[ii,:]=abs(fft)

operators_fft_mul=np.ones([360,4],dtype='complex')
for im,ii in zip(operators_mul,range(360)):
    x,y,z = get_contour(im)
    fft= np.fft.fft(z)[0:4]
    operators_fft_mul[ii,:]=abs(fft)

operators_fft_add=np.ones([360,4],dtype='complex')
for im,ii in zip(operators_add,range(360)):
    x,y,z = get_contour(im)
    fft= np.fft.fft(z)[0:4]
    operators_fft_add[ii,:]=abs(fft)

cov_a=np.cov(operators_fft_minu.T)
cov_b=np.cov(operators_fft_mul.T)
cov_c=np.cov(operators_fft_add.T)

mu_a=np.mean(operators_fft_minu,0)
mu_b=np.mean(operators_fft_mul,0)
mu_c=np.mean(operators_fft_add,0)

def test_opt_bay(imd):
    imd=cv2.cvtColor(imd,cv2.COLOR_BGR2GRAY)
    imd=cv2.resize(imd, (32, 32), interpolation=cv2.INTER_AREA)
    imd=imd/255
    contours = measure.find_contours(imd, 0.5)
    if len(contours)==3:
        return '/'
    elif len(contours)==2:
        return '='
    else:
        xa,ya,za=get_contour(imd)
        fftd = np.fft.fft(za)[0:4]
        fftd = abs(fftd)
        g_complete = [g(fftd, cov_a, mu_a),
                      g(fftd, cov_b, mu_b),
                      g(fftd, cov_c, mu_c)]
        if np.argmax(g_complete)==0:
            return '-'
        if np.argmax(g_complete)==1:
            return '*'
        if np.argmax(g_complete)==2:
            return '+'
