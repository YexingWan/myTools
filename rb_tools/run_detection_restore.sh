#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=-1
MODEL_NAME="ssd_resnet50_v1_fpn"
CHECKPOINT_PATH="${HOME}/newhome/workspace/googleAPI_model_ckpt/${MODEL_NAME}/model.ckpt"
INPUT_W=640
INPUT_H=640
RESTORE_SAVE_DIR="${HOME}/newhome/workspace/googleAPI_model_ckpt/${MODEL_NAME}/restore"
PIPLINE_CONFIG_DIR="${HOME}/newhome/workspace/models_tensorflow/models/research/object_detection/samples/configs"
PIPLINE_CONFIG_FILE="${PIPLINE_CONFIG_DIR}/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config"

echo Restoring model...

python3 ./run_detection_restore.py \
    --model_name "$MODEL_NAME" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --input_shape "${INPUT_H},${INPUT_W}" \
    --restore_path "$RESTORE_SAVE_DIR" \
    --pipeline_config_path "$PIPLINE_CONFIG_FILE"

if test $? -ne 0; then
    echo Restore fail.
else
    echo Restore done.
fi
