#!/bin/bash

OUTPUT_NODE="feature_fusion/Conv_7/BiasAdd,feature_fusion/Conv_8/BiasAdd,feature_fusion/Conv_9/BiasAdd"
LOGDIR="./graph_vis/"
MODEL_NAME="EAST"

python3 restore.py

echo Restoring model...
if test $? -ne 0; then
    echo Restore fail.
    exit 1
fi

echo Freezing model...
RbCli freeze ./restored/ -o $OUTPUT_NODE --logdir $LOGDIR

echo Parse model to SG-IR...
RbCli sg -n $MODEL_NAME --with-json True --with-coeff True ./model.pb


if test "$1" = '--GPU'; then
    echo Generate prototxt for using CUDA...
    RbCli opt "./$MODEL_NAME_sg.pbtxt" "./$MODEL_NAME_sg.h5" ./CUDA.pbtxt
fi

echo Done.
