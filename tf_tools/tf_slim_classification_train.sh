#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="../crquant-slim:..:$PYTHONPATH"
export PYTHONPATH="/home/yx-wan/newhome/workspace/retrainquant:/home/yx-wan/newhome/coco/tools/cocoapi/PythonAPI:$PYTHONPATH"
PBTXT="/home/yx-wan/nas/BenchmarkData/rainbuilder/tf_sgir_1.4.2_kld/resnet_v1_50_gpu/quant/resnet_v50_quant_sg.pbtxt"

python tf_slim_classification_train.py \
    --train_dir /home/yx-wan/newhome/checkpoint/quant/resnet50_v1_50_quant \
    --save_interval_secs 300 \
    --save_summaries_secs 300 \
    --weight_decay 0.0001 \
    --optimizer adam \
    --learning_rate 0.0000001  \
    --learning_rate_decay_factor 0.8 \
    --learning_rate_decay_type exponential \
    --num_epochs_per_decay 1 \
    --dataset_dir /home/yx-wan/newhome/Imagenet/train_no_resize \
    --labels_offset 1 \
    --model_name resnet_v1_50 \
    --batch_size 32 \
    --number_epochs 10 \
    --checkpoint_path ~/newhome/checkpoint/resnet_v1_50/resnet_v1_50.ckpt \
    --ignore_missing_vars True \
    --freeze_bn_epoch 1 \
    --freeze_quant_epoch 8 \
    --end_learning_rate 0 \
    --moving_average_decay 0.99999 \
    --pbtxt $PBTXT \
    --crquant True

#--learning_rate_decay_factor 0.1 \
#--moving_average_decay 0.9999 \
#--label_smoothing 0.4 \
#--max_number_of_steps

#tf.app.flags.DEFINE_string(
#    'master', '', 'The address of the TensorFlow master to use.')
#
#tf.app.flags.DEFINE_string(
#    'train_dir', '/tmp/tfmodel/',
#    'Directory where checkpoints and event logs are written to.')
#
#tf.app.flags.DEFINE_integer('num_clones', 1,
#                            'Number of model clones to deploy. Note For '
#                            'historical reasons loss from all clones averaged '
#                            'out and learning rate decay happen per clone '
#                            'epochs')
#
#tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
#                            'Use CPUs to deploy clones.')
#
#tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')
#
#tf.app.flags.DEFINE_integer(
#    'num_ps_tasks', 0,
#    'The number of parameter servers. If the value is 0, then the parameters '
#    'are handled locally by the worker.')
#
#tf.app.flags.DEFINE_integer(
#    'num_readers', 4,
#    'The number of parallel readers that read data from the dataset.')
#
#tf.app.flags.DEFINE_integer(
#    'num_preprocessing_threads', 4,
#    'The number of threads used to create the batches.')
#
#tf.app.flags.DEFINE_integer(
#    'log_every_n_steps', 10,
#    'The frequency with which logs are print.')
#
#tf.app.flags.DEFINE_integer(
#    'save_summaries_secs', 600,
#    'The frequency with which summaries are saved, in seconds.')
#
#tf.app.flags.DEFINE_integer(
#    'save_interval_secs', 600,
#    'The frequency with which the model is saved, in seconds.')
#
#
#tf.app.flags.DEFINE_integer(
#    'task', 0, 'Task id of the replica running the training.')
#
#tf.app.flags.DEFINE_integer(
#    'eval_image_size', None, 'Eval image size')
#
#######################
## Optimization Flags #
#######################
#
#tf.app.flags.DEFINE_float(
#    'weight_decay', 0.00004, 'The weight decay on the model weights.')
#
#tf.app.flags.DEFINE_string(
#    'optimizer', 'rmsprop',
#    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
#    '"ftrl", "momentum", "sgd" or "rmsprop".')
#
#tf.app.flags.DEFINE_float(
#    'adadelta_rho', 0.95,
#    'The decay rate for adadelta.')
#
#tf.app.flags.DEFINE_float(
#    'adagrad_initial_accumulator_value', 0.1,
#    'Starting value for the AdaGrad accumulators.')
#
#tf.app.flags.DEFINE_float(
#    'adam_beta1', 0.9,
#    'The exponential decay rate for the 1st moment estimates.')
#
#tf.app.flags.DEFINE_float(
#    'adam_beta2', 0.999,
#    'The exponential decay rate for the 2nd moment estimates.')
#
#tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
#
#tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
#                          'The learning rate power.')
#
#tf.app.flags.DEFINE_float(
#    'ftrl_initial_accumulator_value', 0.1,
#    'Starting value for the FTRL accumulators.')
#
#tf.app.flags.DEFINE_float(
#    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
#
#tf.app.flags.DEFINE_float(
#    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
#
#tf.app.flags.DEFINE_float(
#    'momentum', 0.9,
#    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
#
#tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
#
#tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
#
#tf.app.flags.DEFINE_integer(
#    'quantize_delay', -1,
#    'Number of steps to start quantized training. Set to -1 would disable '
#    'quantized training.')
#
########################
## Learning Rate Flags #
########################
#
#tf.app.flags.DEFINE_string(
#    'learning_rate_decay_type',
#    'exponential',
#    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
#    ' or "polynomial"')
#
#tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
#
#tf.app.flags.DEFINE_float(
#    'end_learning_rate', 0.0001,
#    'The minimal end learning rate used by a polynomial decay learning rate.')
#
#tf.app.flags.DEFINE_float(
#    'label_smoothing', 0.0, 'The amount of label smoothing.')
#
#tf.app.flags.DEFINE_float(
#    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
#
#tf.app.flags.DEFINE_float(
#    'num_epochs_per_decay', 2.0,
#    'Number of epochs after which learning rate decays. Note: this flag counts '
#    'epochs per clone but aggregates per sync replicas. So 1.0 means that '
#    'each clone will go over full epoch individually, but replicas will go '
#    'once across all replicas.')
#
#tf.app.flags.DEFINE_bool(
#    'sync_replicas', False,
#    'Whether or not to synchronize the replicas during training.')
#
#tf.app.flags.DEFINE_integer(
#    'replicas_to_aggregate', 1,
#    'The Number of gradients to collect before updating params.')
#
#tf.app.flags.DEFINE_float(
#    'moving_average_decay', None,
#    'The decay to use for the moving average.'
#    'If left as None, then moving averages are not used.')
#
########################
## Dataset Flags #
########################
#
#tf.app.flags.DEFINE_string(
#    'dataset_name', 'imagenet', 'The name of the dataset to load.')
#
#tf.app.flags.DEFINE_string(
#    'dataset_split_name', 'train', 'The name of the train/test split.')
#
#tf.app.flags.DEFINE_string(
#    'dataset_dir', None, 'The directory where the dataset files are stored.')
#
#tf.app.flags.DEFINE_integer(
#    'num_samples', 1281000, 'The number of samples in whole training dataset.')
#
#tf.app.flags.DEFINE_integer(
#    'labels_offset', 0,
#    'An offset for the labels in the dataset. This flag is primarily used to '
#    'evaluate the VGG and ResNet architectures which do not use a background '
#    'class for the ImageNet dataset.')
#
#tf.app.flags.DEFINE_string(
#    'model_name', 'inception_v3', 'The name of the architecture to train.')
#
#tf.app.flags.DEFINE_string(
#    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
#                                'as `None`, then the model_name flag is used.')
#
#tf.app.flags.DEFINE_integer(
#    'batch_size', 32, 'The number of samples in each batch.')
#
#tf.app.flags.DEFINE_integer(
#    'train_image_size', None, 'Train image size')
#
#tf.app.flags.DEFINE_integer('max_number_of_steps', None,
#                            'The maximum number of training steps.')
#
######################
## Fine-Tuning Flags #
######################
#
#tf.app.flags.DEFINE_string(
#    'checkpoint_path', None,
#    'The path to a checkpoint from which to fine-tune.')
#
#tf.app.flags.DEFINE_string(
#    'checkpoint_exclude_scopes', None,
#    'Comma-separated list of scopes of variables to exclude when restoring '
#    'from a checkpoint.')
#
#tf.app.flags.DEFINE_string(
#    'trainable_scopes', None,
#    'Comma-separated list of scopes to filter the set of variables to train.'
#    'By default, None would train all the variables.')
#
#tf.app.flags.DEFINE_boolean(
#    'ignore_missing_vars', False,
#    'When restoring a checkpoint would ignore missing variables.')
#
######################
## Crquant Flags #
######################
#
#tf.app.flags.DEFINE_boolean(
#    'crquant', False, 'Whether do crquant.')
#tf.app.flags.DEFINE_integer('freeze_bn_epoch', 5,
#                            'freeze bn update before number of epoch')
#
#tf.app.flags.DEFINE_integer('freeze_quant_epoch', 6,
#                            'freeze quant layer update before number of epoch')
#
#tf.app.flags.DEFINE_integer('calibration_step', 0,
#                            'calibration step before retraining')
#
#tf.app.flags.DEFINE_integer('batch_size', 24,
#                            'batch size for calibration')
#
#tf.app.flags.DEFINE_string('excluded_scopes', None,
#                           'scope excluded from quant')
#
#tf.app.flags.DEFINE_string('pbtxt', None, 'pbtxt for model')