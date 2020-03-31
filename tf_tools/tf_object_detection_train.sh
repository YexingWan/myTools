#!/usr/bin/env bash

MODEL='ssd_resnet50_v1_fpn'
echo "Training model: ${MODEL}"
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH="../crquant-slim:..:$PYTHONPATH"
export PYTHONPATH="${HOME}/newhome/workspace/retrainquant:${HOME}/newhome/coco/tools/cocoapi/PythonAPI:$PYTHONPATH"



TRAIN_DIR="${HOME}/newhome/checkpoint/${MODEL}/EMA_quant_retrain_0"
PIPLINE_CONFIG_DIR="${HOME}/newhome/workspace/models_tensorflow/models/research/object_detection/samples/configs"
PIPLINE_CONFIG_FILE="${PIPLINE_CONFIG_DIR}/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config"
PBTXT="/nas/BenchmarkData/rainbuilder/tf_sgir_1.4.2_kld/ssd_fpn_gpu/quant/ssd_fpn_quant_sg.pbtxt"



python tf_object_detection_train.py \
--train_dir $TRAIN_DIR \
--pipeline_config_path $PIPLINE_CONFIG_FILE \
--crquant True \
--pbtxt ${PBTXT} \
--freeze_bn_epoch 0 \
--freeze_quant_epoch 1

#flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
#flags.DEFINE_integer('task', 0, 'task id')
#flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
#flags.DEFINE_boolean('clone_on_cpu', False,
#                     'Force clones to be deployed on CPU.  Note that even if '
#                     'set to False (allowing ops to run on gpu), some ops may '
#                     'still be run on the CPU if they have no GPU kernel.')
#flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
#                                           'replicas.')
#flags.DEFINE_integer('ps_tasks', 0,
#                     'Number of parameter server tasks. If None, does not use '
#                     'a parameter server.')
#flags.DEFINE_string('train_dir', '',
#                    'Directory to save the checkpoints and training summaries.')
#
#flags.DEFINE_string('pipeline_config_path', '',
#                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
#                    'file. If provided, other configs are ignored')
#
#flags.DEFINE_string('train_config_path', '',
#                    'Path to a train_pb2.TrainConfig config file.')
#flags.DEFINE_string('input_config_path', '',
#                    'Path to an input_reader_pb2.InputReader config file.')
#flags.DEFINE_string('model_config_path', '',
#                    'Path to a model_pb2.DetectionModel config file.')
#
######################
## Crquant Flags #
######################
#
#flags.DEFINE_boolean(
#    'crquant', False, 'Whether do crquant.')
#flags.DEFINE_integer('freeze_bn_epoch', 5,
#                            'freeze bn update before number of epoch')
#flags.DEFINE_integer('freeze_quant_epoch', 6,
#                            'freeze quant layer update before number of epoch')
#flags.DEFINE_integer('calibration_step', 0,
#                            'calibration step before retraining')
#flags.DEFINE_string('excluded_scopes', None,
#                           'scope excluded from quant')
#flags.DEFINE_string('pbtxt', None, 'pbtxt for model')
#flags.DEFINE_float(
#    'scale_factor', 1.,
#    "scale update factor to model weight while retrain.")
#flags.DEFINE_float(
#    'zero_point_factor', 256. * 256.,
#    "zp update factor to model weight while retrain.")
#flags.DEFINE_string('variables_scope_replace_dict_key', None, 'keys of scope-replace dictionary used in crquant')
#flags.DEFINE_string('variables_scope_replace_dict_key_split', ',', 'split for keys')
#flags.DEFINE_string('variables_scope_replace_dict_value', None, 'values of scope-replace dictionary used in crquant')
#flags.DEFINE_string('variables_scope_replace_dict_value_split', ',', 'split for values')
#
####################################
## Try to  get these from config#
####################################
#
#flags.DEFINE_integer(
#    'num_samples', 1281000, 'The number of samples in whole training dataset.')
