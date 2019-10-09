#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=-1
MODEL_NAME="inception_resnet_v2"
CHECKPOINT_PATH="/home/yx-wan/newhome/checkpoint/${MODEL_NAME}/${MODEL_NAME}.ckpt"
NUM_CLASS=1001
INPUT_W=299
INPUT_H=299
RESTORE_SAVE_DIR="/home/yx-wan/newhome/checkpoint/${MODEL_NAME}/restored"


echo Restoring model...

python3 ./run_restore.py \
    --model_name "$MODEL_NAME" \
    --num_classes "$NUM_CLASS" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --input_shape $INPUT_H $INPUT_W \
    --restore_path "$RESTORE_SAVE_DIR"

if test $? -ne 0; then
    echo Restore fail.
else
    echo Restore done.
fi
