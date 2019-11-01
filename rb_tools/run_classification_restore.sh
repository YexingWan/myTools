#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=-1
MODEL_NAME="ssd_resnet50_v1_fpn"
CHECKPOINT_PATH="${HOME}/newhome/workspace/googleAPI_model_ckpt/${MODEL_NAME}/model.ckpt"
NUM_CLASS=1001
INPUT_W=299
INPUT_H=299
RESTORE_SAVE_DIR="/home/yx-wan/newhome/checkpoint/${MODEL_NAME}/restored"


echo Restoring model...

python3 ./run_classification_restore.py \
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
