#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2

MODEL='inception_resnet_v2'

echo "Quantifying model: ${MODEL}"

OUTPUT_DIR="./${MODEL}"
PBTXT="${MODEL}_sg.pbtxt"
H5="${MODEL}_sg.h5"
mkdir "$OUTPUT_DIR/quant"
IMAGE_DIR='/root/newhome/Imagenet/quant'

RbCli quant $OUTPUT_DIR/$PBTXT \
            $OUTPUT_DIR/$H5 \
            $OUTPUT_DIR/preprocess.py \
            --preprocess-range -1.0,1.0 \
            --img-dir $IMAGE_DIR \
            --with-coeff True \
            --with-sim True \
            --output-dir "$OUTPUT_DIR/quant" \
            --gpu-config ./CUDA.pbtxt
