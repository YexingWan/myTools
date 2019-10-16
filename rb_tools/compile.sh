#!/bin/bash
export CUDA_VISIBLE_DEVICES=-1
OUTPUT_NODE="InceptionResnetV2/Logits/Logits/BiasAdd"
LOGDIR_NAME="graph_vis"
MODEL_NAME="inception_resnet_v2"
RESTORED_DIR="/home/yx-wan/newhome/checkpoint/${MODEL_NAME}/restored"

mkdir "./$MODEL_NAME"

echo Freezing model...
RbCli freeze "$RESTORED_DIR" -o $OUTPUT_NODE --logdir "./$MODEL_NAME" -m "./$MODEL_NAME/$MODEL_NAME.pb"

echo Parse model to SG-IR...
RbCli tf -o "./$MODEL_NAME" -n $MODEL_NAME --with-json True --with-coeff True "./$MODEL_NAME/$MODEL_NAME.pb"


echo Done.
