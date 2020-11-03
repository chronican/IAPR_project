import cv2
import numpy as np
import os
import skimage
from utilities import *

# transform input video into RGB images
def read_video(path,isArtifact):
    if not os.path.exists(path):
        raise Exception('Specified path at {} does not contain a video file!'.format(path))
    cap = cv2.VideoCapture(path)
    im_input = []
    # rot_angle = (np.random.rand()*2-1) * 10 + 90*np.random.randint(0,4)
    rot_angle = 0
    gamma_rand = np.random.randint(0,5)/10+0.8
    # resize_rand = np.random.rand()*0.6+0.7
    resize_rand = 1
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is not None:
            # for robustness test
            if isArtifact:
                frame = skimage.transform.rotate(frame,angle=rot_angle,preserve_range=True).astype('uint8')
                frame = skimage.exposure.adjust_gamma(frame, gamma=gamma_rand, gain=1)
                frame = frame/frame.max()
                frame = skimage.img_as_ubyte(frame)
                frame = skimage.transform.resize(frame,(int(frame.shape[0]*resize_rand),int(frame.shape[1]*resize_rand)),preserve_range=True).astype('uint8')
            frame = aug(frame) # augmentation for better bbox detection
            im_input.append(frame) # read image as BGR format for consistency
        else:
            break
    cap.release()
    im_input = np.asarray(im_input, dtype='uint8') # save as unit8 for consistency
    print('Successfully reading video at {}, with {:d} frames detected.'.format(path, im_input.shape[0]))
    return im_input

# output processed images to a 2-FPS video
def output_video(path,im_output):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2,
                          (im_output[0].shape[1], im_output[0].shape[0]))
    for frame in im_output:
        out.write(frame)
    print('Sucessfully output video at {}.'.format(path))