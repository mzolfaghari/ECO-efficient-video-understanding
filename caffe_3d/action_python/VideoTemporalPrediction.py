'''
A sample function for classification using temporal network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import glob
import os
import numpy as np
import math
import cv2
import scipy.io as sio

def VideoTemporalPrediction(
        vid_name,
        mean_file,
        net,
        num_categories,
        feature_layer,
        start_frame=0,
        num_frames=0,
        num_samples=25,
        optical_flow_frames=10
        ):

    if num_frames == 0:
        imglist = glob.glob(os.path.join(vid_name, '*flow_x*.jpg'))
        duration = len(imglist)
    else:
        duration = num_frames

    # selection
    step = int(math.floor((duration-optical_flow_frames+1)/num_samples))
    dims = (256,340,optical_flow_frames*2,num_samples)
    flow = np.zeros(shape=dims, dtype=np.float64)
    flow_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        for j in range(optical_flow_frames):
            flow_x_file = os.path.join(vid_name, 'flow_x_{0:05d}.jpg'.format(i*step+j+1 + start_frame))
            flow_y_file = os.path.join(vid_name, 'flow_y_{0:05d}.jpg'.format(i*step+j+1 + start_frame))
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            img_x = cv2.resize(img_x, dims[1::-1])
            img_y = cv2.resize(img_y, dims[1::-1])

            flow[:,:,j*2  ,i] = img_x
            flow[:,:,j*2+1,i] = img_y

            flow_flip[:,:,j*2  ,i] = 255 - img_x[:, ::-1]
            flow_flip[:,:,j*2+1,i] = img_y[:, ::-1]

    # crop
    flow_1 = flow[:224, :224, :,:]
    flow_2 = flow[:224, -224:, :,:]
    flow_3 = flow[16:240, 60:284, :,:]
    flow_4 = flow[-224:, :224, :,:]
    flow_5 = flow[-224:, -224:, :,:]
    flow_f_1 = flow_flip[:224, :224, :,:]
    flow_f_2 = flow_flip[:224, -224:, :,:]
    flow_f_3 = flow_flip[16:240, 60:284, :,:]
    flow_f_4 = flow_flip[-224:, :224, :,:]
    flow_f_5 = flow_flip[-224:, -224:, :,:]

    flow = np.concatenate((flow_1,flow_2,flow_3,flow_4,flow_5,flow_f_1,flow_f_2,flow_f_3,flow_f_4,flow_f_5), axis=3)

    # substract mean
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']

    flow = flow - np.tile(flow_mean[...,np.newaxis], (1, 1, 1, flow.shape[3]))
    flow = np.transpose(flow, (1,0,2,3))

    # test
    batch_size = 50
    prediction = np.zeros((num_categories,flow.shape[3]))
    num_batches = int(math.ceil(float(flow.shape[3])/batch_size))

    for bb in range(num_batches):
        span = range(batch_size*bb, min(flow.shape[3],batch_size*(bb+1)))

        net.blobs['data'].data[...] = np.transpose(flow[:,:,:,span], (3,2,1,0))
        output = net.forward()

        prediction[:, span] = np.transpose(output[feature_layer])

    return prediction
