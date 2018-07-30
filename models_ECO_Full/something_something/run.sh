
mkdir -p snapshots
mkdir -p training

HDF5_DISABLE_VERSION_CHECK=1 /home/zolfagha/local/temporal_segmentNet/caffe-3D//build/tools/caffe train --solver=solver.prototxt --weights=/models/bn_inception_kinetics_rgb_pretrained.caffemodel,/models/ECO_Lite_SomethingSomething.caffemodel 2>&1 | tee -a training/log.txt









