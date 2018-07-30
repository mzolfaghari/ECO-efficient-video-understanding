#!/usr/bin/env python

'''
A sample script to run classificition using both spatial/temporal nets.
Modify this script as needed.
'''

import numpy as np
import caffe
import math

from VideoSpatialPrediction import VideoSpatialPrediction
from VideoTemporalPrediction import VideoTemporalPrediction

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def main():

    # caffe init
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    # spatial prediction
    model_def_file = '../models/action_recognition/dextro_spatial.prototxt'
    model_file = '../dextro_benchmark_rgb_iter_48000.caffemodel'
    spatial_net = caffe.Net(model_def_file, model_file, caffe.TEST)

    # temporal prediction
    model_def_file = '../models/action_recognition/dextro_temporal.prototxt'
    model_file = '../dextro_benchmark_flow_iter_39000.caffemodel'
    temporal_net = caffe.Net(model_def_file, model_file, caffe.TEST)

    # input video (containing image_*.jpg and flow_*.jpg) and some settings
    input_video_dir = 'video/'
    start_frame = 0
    num_categories = 131
    feature_layer = 'fc8-2'

    # temporal net prediction
    temporal_mean_file = 'flow_mean.mat'
    temporal_prediction = VideoTemporalPrediction(
            input_video_dir,
            temporal_mean_file,
            temporal_net,
            num_categories,
            feature_layer,
            start_frame)
    avg_temporal_pred_fc8 = np.mean(temporal_prediction, axis=1)
    avg_temporal_pred = softmax(avg_temporal_pred_fc8)

    # spatial net prediction
    spatial_mean_file = 'rgb_mean.mat'
    spatial_prediction = VideoSpatialPrediction(
            input_video_dir,
            spatial_mean_file,
            spatial_net,
            num_categories,
            feature_layer,
            start_frame)
    avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
    avg_spatial_pred = softmax(avg_spatial_pred_fc8)

    # fused prediction (temporal:spatial = 2:1)
    fused_pred = np.array(avg_temporal_pred) * 2./3 + \
                 np.array(avg_spatial_pred) * 1./3

if __name__ == "__main__":
    main()
