"""script for predicting labels from live feed"""

import numpy as np
import caffe
import cv2
import math
import scipy.io as sio
import time
import random
import itertools
batch_size = 16 #number of samples per video

def online_predict(mean_file,model_def_file,model_file,classes_file,num_categories):
    # caffe init
    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()

    frame_counter = 0
    index_to_label = {}
    
    #sampling scheme
    algo = [[16],[8,8],[4,4,8],[2,2,4,8],[1,1,2,4,8]]
    
    with open(classes_file,"r") as file:
      for line in file:
        index, label = line.strip().split(" ",1)
        index_to_label[int(index)] = label
        
    
    net = caffe.Net( model_file,model_def_file, caffe.TEST)
    cap = cv2.VideoCapture(0)
    
    
    dims = (256,340,3,batch_size)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)
    
    #show frame_num(time) and predictions
    text = ""
    time = ""
    
    d = sio.loadmat(mean_file)
    image_mean = d['image_mean']

    running_frames = []
    last_16_frames = []
    initial_predictions = np.zeros((num_categories , 1))

    while(True):
      # Capture frame-by-frame
      time = "Frame: " + str(frame_counter)
      ret, frame = cap.read()

      cv2.putText(frame,text, (10,80),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255) , thickness = 2 )
      cv2.imshow('Frames',frame)
      img = cv2.resize(frame, dims[1::-1])
      
      last_16_frames.append(img)
      if frame_counter == (batch_size * 6):
        frame_counter = 0
      frame_counter = frame_counter + 1      

      if (frame_counter % batch_size == 0):
        
        rgb = np.zeros(shape=dims, dtype=np.float64)
        running_frames.append(last_16_frames)        
        n_slots = len(running_frames)

        if(n_slots>5):
          del running_frames[0]
          frames_algo = algo[4]
        else:
          frames_algo = algo[n_slots-1]
        for y in range(len(frames_algo)):
          idx_frames = np.rint(np.linspace( 0 ,len(running_frames[y]) -1, frames_algo[y] )).astype(np.int16) 
          print(idx_frames)          
          running_frames[y] = [running_frames[y][i] for i in idx_frames]
          
        last_16_frames = []
        flattened_list  = list(itertools.chain(*running_frames))
        for ix,img_arr in enumerate(flattened_list):
          rgb[:,:,:,ix] = img_arr
	
        rgb_3 = rgb[16:240, 60:284, :,:]
        rgb = rgb_3
        rgb = rgb[...] - np.tile(image_mean[...,np.newaxis], (1, 1, 1, rgb.shape[3]))
        rgb = np.transpose(rgb, (1,0,2,3))
        
        prediction = np.zeros((num_categories,1))
        
        net.blobs['data'].data[...] = np.transpose(rgb[:,:,:,:], (3,2,1,0))
        output = net.forward()
        prediction[:, :] = np.transpose(output["fc8"])
        predictions_mean = np.mean(prediction + initial_predictions , axis=1)
        
        initial_predictions = predictions_mean
        predict_ind = np.argmax(predictions_mean)
        
        
        text = "Action: " + index_to_label[int(predict_ind)]
        
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
      
    
if __name__ == "__main__":

    # model files
    mean_file = "../rgb_mean.mat"
    model_def_file = '../Models/ECO_Lite_kinetics.caffemodel'
    model_file = '/models_ECO_Lite/kinetics/deploy.prototxt'
    #class indices file
    classes_file = "../class_ind_kinetics.txt"
    #num_categories
    num_categories = 400
    
    online_predict(mean_file,model_def_file,model_file,classes_file,num_categories)

