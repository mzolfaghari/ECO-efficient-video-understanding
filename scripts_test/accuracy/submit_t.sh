

mkdir -p evalresult


HDF5_DISABLE_VERSION_CHECK=1 /../caffe_3d//build/tools/caffe test  --model=../ECO_full_ucf101.prototxt --weights=../ECO_full_UCF101.caffemodel -gpu 0 -iterations 3783 2>&1 | tee -a evalresult/log.txt









