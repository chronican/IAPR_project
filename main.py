import argparse
import random

from utilities import *
from videoIO import *
from segmentation import *
from classification import *
from class_bay_operator import *

if __name__ == "__main__":
    # set random seed
    # random.seed(2020)

    # parser
    parser = argparse.ArgumentParser(description='Specify input and output path')
    parser.add_argument('--input', default='./data/robot_parcours_1.avi', help='path for input video')
    parser.add_argument('--output', default='./data/output.avi', help='path for output video')
    parser.add_argument('--bayesian',action='store_true',help='use bayesian classification')
    parser.add_argument('--plot',action='store_true', help='plot intermediate results')
    parser.add_argument('--artifact',action='store_true', help='introduce artifacts for robustness test')
    parser.add_argument('--exam',action='store_true',help='disable all robustness test in exam mode')
    opt = parser.parse_args()
    if opt.exam:
        opt.plot = False
        opt.artifact = False

    # read input video, done
    elapsed_beginning = time.time()
    elapsed = time.time()
    print('Start to read video...')
    im_input = read_video(opt.input,opt.artifact)
    elapsed = time.time() - elapsed
    print('Time used: {:.2f} s'.format(elapsed))

    # do processing for the first image, segment for bounding boxes
    print('Start to do processing for bounding box...')
    elapsed = time.time()
    init_im = im_input[0]
    bbox,im_bbox = get_bbox(init_im,opt.plot) # asjusting here

    # for robustness test
    if opt.artifact:
        bbox_elements = []
        for i in range(len(bbox)):
            t = init_im[bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3], :]
            t_gray = cv2.cvtColor(init_im[bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3], :], cv2.COLOR_BGR2GRAY)
            for j in range(3):
                t[:, :, j] = t_gray
            im_input[0,bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3],:] = t
            if i == 9: # to test minus sign
                minus_test = cv2.imread('./data/minus_test.png')
                minus_test = cv2.resize(minus_test, t_gray.shape, interpolation=cv2.INTER_AREA)
                im_input[0,bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3],:] = minus_test.transpose((1,0,2))
        init_im = im_input[0]

    bbox_elements = [basic_elements(index=i,bbox_pos=bbox[i],
                                   BGRimage=init_im[bbox[i][0]:bbox[i][2], bbox[i][1]:bbox[i][3], :],
                                   isDigit=None,isBayesian=opt.bayesian) for i in range(len(bbox))]
    elapsed = time.time() - elapsed
    print('Time used: {:.2f} s'.format(elapsed))

    # train the model if it's not saved in current folder
    print('Start to prepare the neural network...')
    elapsed = time.time()
    if os.path.exists('./nn_operator.h5'):
        model_opr = load_model('./nn_operator.h5')
    else:
        exec(open('train_classifier_operator.py').read())
        model_opr = load_model('./nn_operator.h5')
    if os.path.exists('./nn_digit.h5'):
        model_digit = load_model('./nn_digit.h5')
    else:
        exec(open('train_classifier_digit.py').read())
        model_digit = load_model('./nn_digit.h5')
    elapsed = time.time() - elapsed
    print('Time used: {:.2f} s'.format(elapsed))

    # classification for each bounding box
    print('Start to classify each bounding box...')
    elapsed = time.time()
    for bbox in bbox_elements:
        bbox.detect_symbol(model_digit,model_opr)
    elapsed = time.time() - elapsed
    print('Time used: {:.2f} s'.format(elapsed))
  
    # processing for each image
    print('Start to process each image...')
    im_output = im_input.copy()
    im_output[0] = im_bbox
    last_centerDetc, last_detc = None,None  # center position when a bounding box is detected
    for i in range(len(im_input)):
        elapsed = time.time()
        cur_im = im_output[i]
        cur_center = get_arrow_pos(cur_im)
        # update robot trajectory
        if i == 0: # assume in the first frame, the robot is not above any element
            our_robot = robot(init_pos=cur_center, init_detc=None,bbox_elements=bbox_elements,\
                              model_digit=model_digit,model_opr=model_opr,total_frame=len(im_input))
            our_robot.process_bbox_info()
        else:
            our_robot.cur_pos = cur_center
            our_robot.hist_pos.append(our_robot.cur_pos)
            our_robot.detect_bbox()
        if i == len(im_input)-1:
            cur_im = our_robot.update_formula(cur_im,frame_id='last', isPlot=opt.plot)
        else:
            cur_im = our_robot.update_formula(cur_im, frame_id=i, isPlot=opt.plot)
        elapsed = time.time() - elapsed
        print('Processed image #{:d}/{:d}, time used: {:.2f} s'.format(i,len(im_input),elapsed))
    print('Equation detected is:'+our_robot.formula)

    print('Start to output video...')
    elapsed = time.time()
    output_video(opt.output, im_output)
    elapsed = time.time() - elapsed
    print('Time used: {:.2f} s'.format(elapsed))

    elapsed = time.time() - elapsed_beginning
    print('Done without error! Total time used: {:.2f} s'.format(elapsed))
