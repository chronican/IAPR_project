import io
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from numpy import *
from collections import Counter
import time
import skimage.transform
from classification import *
from class_bay_operator import *

# class for static basic elements (digits and operators)
class basic_elements():
    def __init__(self,index,bbox_pos,BGRimage,isDigit=None,isBayesian=False):
        self.id = index # integer index
        self.bbox_pos = bbox_pos # a list for bounding box position, minr minc maxr maxc
        self.image = BGRimage.astype('uint8') # save BGR images with uint8 data type
        if isDigit == None: # if not specified, use the default discrimination via color (ignore bonus question)
            self.isDigit = self.get_isDigit()
        self.symbol = None # a string for true symbol, such as '0','1','2'... or '=','+','/','*'
        self.isBayesian = isBayesian

    # to tell by color, if an element is digit or operator
    def get_isDigit(self):
        lower_black = np.array([0, 0, 0]) # hyperparamter here!
        upper_black = np.array([180, 255, 46]) # hyperparamter here!
        img_ = self.image.copy()
        img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2HSV)
        mask_digit = cv2.inRange(img_, lower_black, upper_black)
        if np.sum(mask_digit) > 0:
            return True
        else:
            return False

    # detect the symbol of each bounding box
    def detect_symbol(self,model_digit,model_opr):
        if self.isDigit == True:
            symbol_est = []
            for angle in range(-180, 190, 5):
                im_test = skimage.transform.rotate(self.image, angle=angle, preserve_range=True)
                symbol_est.append(test_digit(im_test.astype('uint8'), model_digit))
            symbol_chosen = most_frequent(symbol_est)
            self.symbol = symbol_chosen
            print('Bbox {} is {}'.format(self.id, self.symbol))
        else:
            if self.isBayesian:
                self.symbol = test_opt_bay(self.image.astype('uint8'))
            else:
                symbol_est = []
                for angle in range(-180, 190, 5):
                    im_test = skimage.transform.rotate(self.image, angle=angle, preserve_range=True)
                    symbol_est.append(test_opt(im_test.astype('uint8'), model_opr))
                symbol_chosen = most_frequent(symbol_est)
                self.symbol = symbol_chosen
            print('Bbox {} is {}'.format(self.id, self.symbol))

    # re-detect the symbol of each bounding box by forceful selection of detection method
    def re_detect_symbol(self,model_digit,model_opr,mode):
        if mode == 'digit':
            self.isDigit = True
            symbol_est = []
            for angle in range(-180, 190, 5):
                im_test = skimage.transform.rotate(self.image, angle=angle, preserve_range=True)
                symbol_est.append(test_digit(im_test.astype('uint8'), model_digit))
            symbol_chosen = most_frequent(symbol_est)
            self.symbol = symbol_chosen
            print('Re-decetion: bbox {} is {}'.format(self.id, self.symbol))
        elif mode == 'operator':
            if self.isBayesian:
                self.symbol = test_opt_bay(self.image.astype('uint8'))
            else:
                self.isDigit = False
                symbol_est = []
                for angle in range(-180, 190, 5):
                    im_test = skimage.transform.rotate(self.image, angle=angle, preserve_range=True)
                    symbol_est.append(test_opt(im_test.astype('uint8'), model_opr))
                symbol_chosen = most_frequent(symbol_est)
                self.symbol = symbol_chosen
            print('Re-decetion: bbox {} is {}'.format(self.id, self.symbol))
        elif mode == 'operator_notEql':
            self.isDigit = False
            symbol_est = []
            for angle in range(-180, 190, 5):
                im_test = skimage.transform.rotate(self.image, angle=angle, preserve_range=True)
                est = test_opt(im_test.astype('uint8'), model_opr)
                if est != '=':
                    symbol_est.append(est)
            symbol_chosen = most_frequent(symbol_est)
            self.symbol = symbol_chosen
            print('Re-decetion: bbox {} is {}'.format(self.id, self.symbol))
        else:
            raise Exception ('Wrong mode input! It should be either `digit`, `operator` or \
                             `operator_notEql`, but it is {} now!'.format(mode))

# class for dynamic trajectory
class robot:
    def __init__(self,bbox_elements,init_pos,model_digit,model_opr,total_frame,init_detc=None):
        self.bbox_elements = bbox_elements  # the bouding box list with their symbols
        self.cur_pos = init_pos # a 2-dim list for current position
        self.hist_pos = [self.cur_pos] # history position including current one
        self.cur_detc = init_detc # a string for detected symbol, such as '0','1','2'... or '=','+','/','*'
        self.hist_detc = [self.cur_detc] # history detection including current one

        self.formula = [] # robot states printed in the image
        self.last_center_detc = None # position at which the last element is detected
        self.last_detc = None # the latest detected element
        self.bbox_center = None # self-evident
        self.bbox_median_size = None # self-evident
        self.bbox_median_dist = None # self-evident
        self.bbox_min_dist = None # self-evident

        self.model_digit = model_digit # self-evident
        self.model_opr = model_opr # self-evident
        self.total_frame = total_frame # self-evident
        self.current_frame = 0

    # process the bounding box position info. to get some key thresholds
    def process_bbox_info(self):
        bbox = [self.bbox_elements[i].bbox_pos for i in range(len(self.bbox_elements))]
        bbox_center = np.asarray(bbox)
        bbox_center = np.array(
            [(bbox_center[:, 0] + bbox_center[:, 2]) / 2, (bbox_center[:, 1] + bbox_center[:, 3]) / 2]).transpose()
        self.bbox_center = bbox_center

        bbox_size = np.asarray(bbox)
        bbox_size = np.array([(bbox_size[:, 2] - bbox_size[:, 0]), (bbox_size[:, 3] - bbox_size[:, 1])]).transpose()
        self.bbox_median_size = np.median(bbox_size.ravel())

        distance = np.zeros([len(bbox),len(bbox)])
        for i in range(len(bbox)):
            for j in range(len(bbox)):
                distance[i,j] = np.linalg.norm(bbox_center[i,:] - bbox_center[j,:],2).sum()
        self.bbox_median_dist = np.median(distance.ravel())
        self.bbox_min_dist = np.min(distance.ravel())

    # to detect which bounding box is touched in the robot's movement, by trajectory
    def detect_bbox(self):
        last_center,cur_center = self.hist_pos[-2],self.cur_pos
        p1, p2 = np.asarray([cur_center[1], cur_center[0]]), np.asarray([last_center[1], last_center[0]])
        # if current position and last position is too close, there cannot be a new element between them
        if self.last_center_detc is not None:
            d_curPrev = np.linalg.norm(np.asarray(cur_center) - self.last_center_detc, 2).sum()
        else:
            d_curPrev = 1e6

        # begin search
        if d_curPrev < 0.75 * self.bbox_min_dist:  # hyperparmeter here!
            cur_bbox_id = None
        else:
            minr, maxr = min(cur_center[0], last_center[0]), max(cur_center[0], last_center[0])
            minc, maxc = min(cur_center[1], last_center[1]), max(cur_center[1], last_center[1])
            d_p2l = []  # distance of the bbox center to the last linear segment of trajectory
            for p3 in self.bbox_center:
                # if the bbox center is too far, ignore this bbox
                if p3[1] < minr - self.bbox_median_size or p3[1] > maxr + self.bbox_median_size or \
                        p3[0] < minc - self.bbox_median_size or p3[0] > maxc + self.bbox_median_size:  # hyperparmeter here!
                    d_p2l.append(1e6)
                else:
                    d_p2l.append(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
            d_p2l = np.asarray(d_p2l)
            cur_bbox_id = np.argmin(d_p2l)
            cur_bbox_id = cur_bbox_id if d_p2l[cur_bbox_id] != 1e6 else None
            # to avoid consecutive duplicate!
            if cur_bbox_id is not None:
                if self.bbox_elements[cur_bbox_id].symbol == self.last_detc and \
                        (d_curPrev < self.bbox_min_dist or np.linalg.norm(p2 - p1, 2).sum() < 2.5*self.bbox_median_size):
                    # hyper-paramters here!
                    cur_bbox_id = None

        if cur_bbox_id is not None:
            # dealing with the first element and the '=' 2 special cases first
            if self.last_detc == None: # for the first passed-above element, it must be a digit!
                if self.bbox_elements[cur_bbox_id].symbol in ['+', '-', '*', '/', '=']: # problematic situation
                    self.bbox_elements[cur_bbox_id].re_detect_symbol(\
                        self.model_digit, self.model_opr, 'digit')
            elif self.bbox_elements[cur_bbox_id].symbol == '=' and \
                        (self.current_frame / self.total_frame) < 0.7:
                # if '=' is detected in the first 50% frames, force redection for a non 'eq' symbol
                self.bbox_elements[cur_bbox_id].re_detect_symbol( \
                    self.model_digit, self.model_opr, 'operator_notEql')

            # normal cases, no consecutive 2 digits or 2 operators
            if self.last_detc in ['+', '-', '*', '/', '=']:
                # if last one is a normal operator or this one must be digit!
                if self.bbox_elements[cur_bbox_id].symbol in ['+', '-', '*', '/', '=']:
                    self.bbox_elements[cur_bbox_id].re_detect_symbol( \
                        self.model_digit, self.model_opr, 'digit')
            elif self.last_detc in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                # last one is digit, this one must be operator!
                if self.bbox_elements[cur_bbox_id].symbol in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                    self.bbox_elements[cur_bbox_id].re_detect_symbol( \
                        self.model_digit, self.model_opr, 'operator')

            # if a '=' has been detected in the last 30% frames, then assume the equation is done!
            if '=' in self.hist_detc and (self.current_frame/self.total_frame) > 0.7\
                and len(self.formula)%2 == 0:
                print('Equation detection is finished in advance, it is {}'.format(self.formula))
            else:
                self.last_center_detc = cur_center
                self.last_detc = self.bbox_elements[cur_bbox_id].symbol
                self.cur_detc = self.bbox_elements[cur_bbox_id].symbol
                self.hist_detc.append(self.cur_detc)
                print('Detected bbox {:d}, it it {}.'.format(cur_bbox_id, self.cur_detc))
        else:
            self.cur_detc = None
            self.hist_detc.append(self.cur_detc)

    def update_formula(self,cur_im,frame_id,isPlot):
        # restore the state of the formula
        self.current_frame += 1
        cur_text = []
        for j in range(len(self.hist_detc)):
            if self.hist_detc[j] is not None:
                # a naive way to reduce duplicate detection
                # no chance of consecutive 2 operators
                cur_text.append(self.hist_detc[j])
        cur_text_final = []
        for j in range(len(cur_text)):
            if j > 1 and cur_text[j] in ['+', '-', '*', '/', '='] \
                    and cur_text[j - 1] in ['+', '-', '*', '/', '=']:
                continue
            else:
                cur_text_final.append(cur_text[j])
        cur_text_final = ''.join(cur_text_final)
        self.formula = cur_text_final
        # annotate output image
        cv2.circle(cur_im,center=self.cur_pos,radius=10,color=(0,255,0),thickness=-1)
        for j in range(len(self.hist_pos)-1):
            cv2.line(cur_im, self.hist_pos[j], self.hist_pos[j+1], (0, 0, 255),thickness=5)

        if frame_id == 'last':
            if cur_text_final[-1] == '=' and cur_text_final.count('=') == 1:
                result = eval(cur_text_final[:-1])
            elif cur_text_final.count('=') == 0:
                cur_text_final = cur_text_final[:-1] + '='
                result = eval(cur_text_final[:-1])
            else:
                    result = ''
            cv2.putText(cur_im, 'Equation=' + cur_text_final + str(result), (0, int(cur_im.shape[1]*0.05)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=2)
        else:
            cv2.putText(cur_im, 'Equation=' + cur_text_final, (0, int(cur_im.shape[1]*0.05)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(255, 0, 0), thickness=2)
        # comment plot in final version
        if isPlot:
            plt.imshow(cv2.cvtColor(cur_im,cv2.COLOR_BGR2RGB))
            plt.tight_layout(h_pad=0,w_pad=0)
            plt.show()
            plt.title('Image #{}'.format(frame_id))

        return cur_im

# rotate images with proper background color
def rotate(img, degree):
    height, width = img.shape[:2]
    heightNew = int(width * np.fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * np.fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(1, 1, 1))
    return imgRotation

# find the most frequent element in a list
def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def compute(img, min_percentile, max_percentile):
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel

#adjust lightness
def aug(src):
    outd = np.zeros(src.shape, src.dtype)
    if get_lightness(src)>130:
        outd=src
    else:
        max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)
        src[src>=max_percentile_pixel] = max_percentile_pixel
        src[src<=min_percentile_pixel] = min_percentile_pixel
        cv2.normalize(src, outd, 255*0.1,255*0.9,cv2.NORM_MINMAX)

    return outd

def get_lightness(src):
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:,:,2].mean()
    
    return  lightness
#adjust white balance
def white_balance(img):
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
 
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
 
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([r, g, b])
    return balance_img
