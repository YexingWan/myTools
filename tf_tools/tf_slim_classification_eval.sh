#!/usr/bin/env bash


MODEL='inception_resnet_v2'
echo "Evaluating model: ${MODEL}"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="../crquant-slim:..:$PYTHONPATH"
export PYTHONPATH="${HOME}/newhome/workspace/retrainquant:${HOME}/newhome/coco/tools/cocoapi/PythonAPI:$PYTHONPATH"
PBTXT="${HOME}/newhome/workspace/myTools/rb_tools/${MODEL}/quant/${MODEL}_quant_sg.pbtxt"
#PBTXT="/nas/BenchmarkData/rainbuilder/tf_sgir_1.4.2_kld/resnet_v1_50_gpu/quant/resnet_v50_quant_sg.pbtxt"

#CHECKPOINT_PATH="${HOME}/newhome/checkpoint/${MODEL}/${MODEL}.ckpt"
#CHECKPOINT_PATH="${HOME}/newhome/checkpoint/quant/resnet50_EMA_train/"
CHECKPOINT_PATH="${HOME}/newhome/checkpoint/${MODEL}/EMA_quant_retrain"
#CHECKPOINT_PATH="${HOME}/newhome/checkpoint/resnet_v1_50/EMA_quant_retrain"



#EVAL_DIR="${HOME}/newhome/checkpoint/${MODEL}/eval_quant_log"
DATASET="${HOME}/newhome/Imagenet/val_no_resize"
EVAL_DIR="${HOME}/newhome/checkpoint/${MODEL}/EMA_quant_retrain/eval_log"
#EVAL_DIR="${HOME}/newhome/checkpoint/quant/resnet50_EMA_calibration_1000/eval_log"



python tf_slim_classification_eval.py \
--batch_size 8 \
--checkpoint_path ${CHECKPOINT_PATH} \
--model_name ${MODEL} \
--preprocessing_name ${MODEL} \
--dataset_dir ${DATASET} \
--eval_dir ${EVAL_DIR} \
--wait_for_checkpoints True \
--excluded_scopes 'InceptionResnetV2/AuxLogits/' \
--crquant True


#--scale_factor 100
#--excluded_scopes 'InceptionResnetV2/AuxLogits/' \
#--scale_factor 100 \
#--pbtxt ${PBTXT} \
#--scale_factor 100 \
#--excluded_scopes 'InceptionResnetV2/AuxLogits/' \
#--pbtxt ${PBTXT} \
#--labels_offset 1 \
#--scale_factor 100 \
#--ignore_missing_vars True \
#--eval_dir ${HOME}/newhome/workspace/myTools/rb_tools/${MODEL}/quant/eval_quant_event \
#--excluded_scopes 'InceptionV3/AuxLogits' \
#--ignore_missing_vars True \
#--pbtxt $PBTXT \





#
#
#--pbtxt $PBTXT \
#--crquant True \
#--pbtxt $PBTXT \
#--wait_for_checkpoints True


#--moving_average_decay 0.9999
#--wait_for_checkpoints True \
#--ignore_missing_vars False \
#--moving_average_decay 0.9999 \
#--pbtxt $PBTXT \
#--crquant True \
#--moving_average_decay 0.9999 \
#--moving_average_decay 0.9999

# from models/research/slim/preprocessing/preprocessing_factory
#  preprocessing_fn_map = {
#      'cifarnet': cifarnet_preprocessing,
#      'inception': inception_preprocessing,
#      'inception_v1': inception_preprocessing,
#      'inception_v2': inception_preprocessing,
#      'inception_v3': inception_preprocessing,
#      'inception_v4': inception_preprocessing,
#      'inception_resnet_v2': inception_preprocessing,
#      'lenet': lenet_preprocessing,
#      'lenet': lenet_preprocessing,
#      'mobilenet_v1': inception_preprocessing,
#      'mobilenet_v2': inception_preprocessing,
#      'mobilenet_v2_035': inception_preprocessing,
#      'mobilenet_v2_140': inception_preprocessing,
#      'nasnet_mobile': inception_preprocessing,
#      'nasnet_large': inception_preprocessing,
#      'pnasnet_mobile': inception_preprocessing,
#      'pnasnet_large': inception_preprocessing,
#      'resnet_v1_50': vgg_preprocessing,
#      'resnet_v1_101': vgg_preprocessing,
#      'resnet_v1_152': vgg_preprocessing,
#      'resnet_v1_200': vgg_preprocessing,
#      'resnet_v2_50': vgg_preprocessing,
#      'resnet_v2_101': vgg_preprocessing,
#      'resnet_v2_152': vgg_preprocessing,
#      'resnet_v2_200': vgg_preprocessing,
#      'vgg': vgg_preprocessing,
#      'vgg_a': vgg_preprocessing,
#      'vgg_16': vgg_preprocessing,
#      'vgg_19': vgg_preprocessing,
#  }



# from models/research/slim/net/net_factory
# networks_map = {'alexnet_v2': alexnet.alexnet_v2,
#                'cifarnet': cifarnet.cifarnet,
#                'overfeat': overfeat.overfeat,
#                'vgg_a': vgg.vgg_a,
#                'vgg_16': vgg.vgg_16,
#                'vgg_19': vgg.vgg_19,
#                'inception_v1': inception.inception_v1,
#                'inception_v2': inception.inception_v2,
#                'inception_v3': inception.inception_v3,
#                'inception_v4': inception.inception_v4,
#                'inception_resnet_v2': inception.inception_resnet_v2,
#                'i3d': i3d.i3d,
#                's3dg': s3dg.s3dg,
#                'lenet': lenet.lenet,
#                'resnet_v1_50': resnet_v1.resnet_v1_50,
#                'resnet_v1_101': resnet_v1.resnet_v1_101,
#                'resnet_v1_152': resnet_v1.resnet_v1_152,
#                'resnet_v1_200': resnet_v1.resnet_v1_200,
#                'resnet_v2_50': resnet_v2.resnet_v2_50,
#                'resnet_v2_101': resnet_v2.resnet_v2_101,
#                'resnet_v2_152': resnet_v2.resnet_v2_152,
#                'resnet_v2_200': resnet_v2.resnet_v2_200,
#                'mobilenet_v1': mobilenet_v1.mobilenet_v1,
#                'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_075,
#                'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
#                'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,
#                'mobilenet_v2': mobilenet_v2.mobilenet,
#                'mobilenet_v2_140': mobilenet_v2.mobilenet_v2_140,
#                'mobilenet_v2_035': mobilenet_v2.mobilenet_v2_035,
#                'nasnet_cifar': nasnet.build_nasnet_cifar,
#                'nasnet_mobile': nasnet.build_nasnet_mobile,
#                'nasnet_large': nasnet.build_nasnet_large,
#                'pnasnet_large': pnasnet.build_pnasnet_large,
#                'pnasnet_mobile': pnasnet.build_pnasnet_mobile,
#               }
