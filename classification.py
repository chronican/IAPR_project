import cv2
from keras.models import load_model
import numpy as np

def test_opt(image,model_opr):
    oud=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    oud = cv2.resize(oud, (32, 32), interpolation=cv2.INTER_AREA)
    data = oud.reshape(1, 32, 32, 1)
    out = model_opr.predict_classes(data, batch_size=1, verbose=0)
    if out == 0:
        return '/'
    if out == 1:
        return '='
    if out == 2:
        return '-'
    if out == 3:
        return '*'
    if out == 4:
        return '+'

def test_digit(image,model_digit):
    dit_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, dit_image = cv2.threshold(dit_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dit_image= np.where((dit_image >= 0) & (dit_image <= 80), 1, 0)
    dit_image= np.array(dit_image).astype("float32")
    dit_image = cv2.resize(dit_image, (28, 28), interpolation=cv2.INTER_AREA)
    data = dit_image.reshape(1, 28, 28, 1)
    out = model_digit.predict_classes(data, batch_size=1, verbose=0)
    return str(out[0])
