#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="../crquant-slim:..:$PYTHONPATH"
export PYTHONPATH="/home/yx-wan/newhome/workspace/retrainquant:/home/yx-wan/newhome/coco/tools/cocoapi/PythonAPI:$PYTHONPATH"
PBTXT="/home/yx-wan/nas/BenchmarkData/rainbuilder/tf_sgir_1.4.2_kld/resnet_v1_50_gpu/quant/resnet_v50_quant_sg.pbtxt"


python tf_slim_classification_eval.py \
--batch_size 8 \
--checkpoint_path  /home/yx-wan/newhome/checkpoint/quant/resnet50_EMA_train \
--model_name resnet_v1_50 \
--preprocessing_name resnet_v1_50 \
--dataset_dir /home/yx-wan/newhome/Imagenet/val_no_resize \
--labels_offset 1 \
--eval_dir /home/yx-wan/newhome/checkpoint/quant/resnet50_EMA_train/eval_no_ema_log \
--crquant True \
--pbtxt $PBTXT \
--wait_for_checkpoints True \
--gpu_memory_fraction 0.3

#--pbtxt $PBTXT \
#--moving_average_decay 0.9999
#--wait_for_checkpoints True \
#--ignore_missing_vars False \
#--moving_average_decay 0.9999 \
#--wait_for_checkpoints True
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