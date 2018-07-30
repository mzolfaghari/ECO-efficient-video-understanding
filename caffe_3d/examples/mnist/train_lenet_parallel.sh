#!/usr/bin/env sh
export LD_LIBRARY_PATH=/usr/local/openmpi/lib/:$LD_LIBRARY_PATH
/usr/local/openmpi/bin/mpirun -np 2 cmake_build/install/bin/caffe train --solver=examples/mnist/lenet_parallel_solver.prototxt
