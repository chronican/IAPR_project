from utilities import *
from skimage import morphology
from skimage import measure
import matplotlib.pyplot as plt
import utilities

# get bounding box, from BGR images
def get_bbox(im_input,isPlot):
    # initialization
    im_bbox = im_input.copy()
    im_gray = cv2.cvtColor(im_input, cv2.COLOR_BGR2GRAY)
    # get segmented regions
    res_rate = (im_input.shape[0]/480.0,im_input.shape[1]/720.0)
    ret, thresh = cv2.threshold(im_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # a = morphology.erosion(thresh, morphology.square(int(16*res_rate[0]*res_rate[1]))) # hyperparamter here!
    a = morphology.erosion(thresh, morphology.square(16))
    b = a.copy()
    label = measure.label(a < 40, connectivity=2) # hyperparamter here!
    borders = np.logical_xor(a, b)
    label[borders] = -1
    regions = measure.regionprops(label)
    # get bounding box position
    bbox = []
    isOutlier = False
    for region, i in zip(measure.regionprops(label), range(len(regions))):
        if region.area > int(3000*res_rate[0]*res_rate[1]) or \
                region.area < int(300*res_rate[0]*res_rate[1]): # hyperparamter here!
            continue
        minr, minc, maxr, maxc = region.bbox
        if (maxc - minc) / (maxr - minr) < 0.2: # remove outlier
            outlier = region.bbox
            isOutlier = True
            continue
        bbox.append(region.bbox)
    # clean up the bbox
    if isOutlier:
        bbox_clean = []
        for cur_bbox in bbox:
            minr, minc, maxr, maxc = cur_bbox
            if cur_bbox[3] < outlier[3] and cur_bbox[1] > outlier[1]: # remove this one
                print('An outlier bbox is found!')
            else:
                bbox_clean.append(cur_bbox)
    else:
        bbox_clean = bbox.copy()
    # print cleaned-up bbox
    for cur_bbox,i in zip(bbox_clean,range(len(bbox_clean))):
        minr, minc, maxr, maxc = cur_bbox
        cv2.rectangle(im_bbox,(minc,minr),(maxc,maxr),color=(0,0,255),thickness=2)
        cv2.putText(im_bbox,str(i),(minc,minr),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,
                    color=(255,0,0),thickness=2)
    print('Successfully detecting {:d} bounding box.'.format(len(bbox_clean)))
    # comment plot in final version
    if isPlot:
        fig = plt.figure(figsize=(8, 6),frameon=False)
        plt.axis('off')
        plt.imshow(cv2.cvtColor(im_bbox,cv2.COLOR_BGR2RGB))
        fig.tight_layout(pad=0)
        plt.show()
        fig.savefig('./data/bbox.png')
        for cur_bbox,ii in zip(bbox_clean,range(len(bbox_clean))):
            minr, minc, maxr, maxc = cur_bbox
            cur_im = im_input[minr:maxr,minc:maxc,:]
            cur_im = cv2.cvtColor(cur_im,cv2.COLOR_BGR2RGB)
            skimage.io.imsave('./data/bbox_{}.png'.format(ii),cur_im)
    return bbox_clean,im_bbox

# get arrow position based on RGB image
def get_arrow_pos(ma):
    redLower = np.array([120, 120, 120]) # hyperparamter here!
    redUpper = np.array([255, 255, 255]) # hyperparamter here!
    hsv_im = cv2.cvtColor(ma, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_im, redLower, redUpper)
    mask = np.array(mask)

    np.where((mask < 80) & (mask > 0), 1, 0) # hyperparamter here!
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] # hyperparamter here!
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return center
