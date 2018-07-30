

mkdir -p snapshots
mkdir -p training


HDF5_DISABLE_VERSION_CHECK=1 ../caffe_3d/build/tools/caffe train --solver=solver.prototxt --weights=/models/ECO_Lite_kinetics.caffemodel 2>&1 | tee -a training/log.txt

