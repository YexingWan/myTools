#!/usr/bin/env bash


MODEL_NAME="ssd_resnet50_v1_fpn"

echo "Evaluating model: ${MODEL_NAME}"
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="../crquant-slim:..:$PYTHONPATH"
export PYTHONPATH="${HOME}/newhome/workspace/retrainquant:${HOME}/newhome/coco/tools/cocoapi/PythonAPI:$PYTHONPATH"



CHECKPOINT_PATH="${HOME}/newhome/workspace/googleAPI_model_ckpt/${MODEL_NAME}"
#CHECKPOINT_PATH="${HOME}/newhome/checkpoint/${MODEL_NAME}/EMA_quant_retrain_0/"
#EVAL_DIR="${HOME}/newhome/workspace/googleAPI_model_ckpt/${MODEL_NAME}/eval_quant_log"
EVAL_DIR="${HOME}/newhome/checkpoint/${MODEL_NAME}/EMA_quant_retrain_0/eval_log"

PIPLINE_CONFIG_DIR="${HOME}/newhome/workspace/models_tensorflow/models/research/object_detection/samples/configs"
PIPLINE_CONFIG_FILE="${PIPLINE_CONFIG_DIR}/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config"
PBTXT="/nas/BenchmarkData/rainbuilder/tf_sgir_1.4.2_kld/ssd_fpn_gpu/quant/ssd_fpn_quant_sg.pbtxt"

python tf_object_detection_eval.py \
--checkpoint_dir $CHECKPOINT_PATH \
--eval_dir $EVAL_DIR \
--pipeline_config_path $PIPLINE_CONFIG_FILE \

#
#
#--crquant True \
#--pbtxt ${PBTXT}






#--run_once True \
#flags = tf.app.flags
#flags.DEFINE_boolean('eval_training_data', False,
#                     'If training data should be evaluated for this job.')
#flags.DEFINE_string(
#    'checkpoint_dir', '',
#    'Directory containing checkpoints to evaluate, typically '
#    'set to `train_dir` used in the training job.')
#flags.DEFINE_string('eval_dir', '', 'Directory to write eval summaries to.')
#flags.DEFINE_string(
#    'pipeline_config_path', '',
#    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
#    'file. If provided, other configs are ignored')
#flags.DEFINE_string('eval_config_path', '',
#                    'Path to an eval_pb2.EvalConfig config file.')
#flags.DEFINE_string('input_config_path', '',
#                    'Path to an input_reader_pb2.InputReader config file.')
#flags.DEFINE_string('model_config_path', '',
#                    'Path to a model_pb2.DetectionModel config file.')
#flags.DEFINE_boolean(
#    'run_once', False, 'Option to only run a single pass of '
#    'evaluation. Overrides the `max_evals` parameter in the '
#    'provided config.')
#FLAGS = flags.FLAGS