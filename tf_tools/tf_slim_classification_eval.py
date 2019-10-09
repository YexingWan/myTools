# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import sys,os

import math
import numpy as np
import tensorflow as tf
import tqdm
from functools import partial
from tf_print import print_configuration_op, print_variables

from re import match

# from datasets import dataset_factory
from tf_record_tools import load_record_classification_ILSVRC_val_dataset

sys.path.append("../models_tensorflow/models/research/slim")
# from models/research/slim
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
    'pbtxt', None, 'The path to pbtxt.')


tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_string(
    'excluded_scopes', None, 'The exclued_scope for crquant. splite by ,')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'crquant', False, 'whether to use quantized graph or not.')


tf.app.flags.DEFINE_bool(
    'wait_for_checkpoints', False, 'whether use evaluation_loop.')

tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 1, 'Fraction using gpu.')

tf.app.flags.DEFINE_bool(
    'ignore_missing_vars', False, 'whether ignore missing vars in checkpoints.')

tf.app.flags.DEFINE_float(
    'scale_factor', 1.,
    "scale update factor to model weight while retrain.")

tf.app.flags.DEFINE_float(
    'zero_point_factor', 256. * 256.,
    "zp update factor to model weight while retrain.")



FLAGS = tf.app.flags.FLAGS

def _get_quant_variable_from_model():
    quant_vars = {}
    for var in tf.all_variables():
        if match(r"^.*\/zero_point:[0-9]*$",var.name) or match(r"^.*\/scale:[0-9]*$",var.name):
            print("quant var:{}".format(var.name))
            quant_vars[var.name[:-2]] = var
    return quant_vars




def main(_):
    print_configuration_op(FLAGS)

    if not os.path.exists(FLAGS.eval_dir):
        os.mkdir(FLAGS.eval_dir)


    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)
    # sess = tf.Session(config=config)
    g = tf.Graph()
    with g.as_default():

        #########################
        # build network function#
        #########################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(1001 - FLAGS.labels_offset),
            is_training=False)
        input_size = FLAGS.eval_image_size or network_fn.default_image_size

        # create global_step to stop crquant create input/quant/global_step: it is a bug of crquant
        tf_global_step = slim.get_or_create_global_step()

        #########################################
        # build dataset and preprocessing module#
        #########################################

        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)
        image_preprocessing_fn = partial(image_preprocessing_fn,
                                         output_height = input_size,
                                         output_width = input_size)
        def preprocess(image, label):
            return image_preprocessing_fn(image), label - FLAGS.labels_offset


        dataset = load_record_classification_ILSVRC_val_dataset(FLAGS.dataset_dir)
        dataset = dataset.map(preprocess)
        dataset = dataset.batch(FLAGS.batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        ######################
        # build prediction op#
        ######################

        # input_place_holder = tf.placeholder(tf.float32,(None,input_size,input_size,3),"input")
        logits, _ = network_fn(images)
        predictions = tf.argmax(logits, 1)
        # now origin graph is constructed


        #############
        # apply ema #
        #############
        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay)
            vars = variable_averages.variables_to_restore(slim.get_model_variables())
            # print("loaded:\n" + "\n\t".join([v for v in vars]))
            # saver = tf.train.Saver(vars)
        else:
            # ema.variables_to_restore will generate proper dictionary base one global_variables
            # so we need to use global_varibales here.
            vars = tf.global_variables()
            # print("loaded:\n" + "\n\t".join([v.name for v in vars]))



        if FLAGS.max_num_batches:
            num_batches = FLAGS.max_num_batches
        else:
            # This ensures that we make a single pass over all of the data.
            num_batches = math.ceil(50000 / float(FLAGS.batch_size))

        # initialize all variable
        # init = tf.global_variables_initializer()
        # sess.run(init)

        ############################
        # add quant node in network#
        ############################
        if FLAGS.crquant:
            import crquant
            crquant.create_graph(
                pbtxt_path=FLAGS.pbtxt,
                is_training=False,
                zero_point_factor=FLAGS.zero_point_factor,
                scale_factor=FLAGS.scale_factor,
                excluded_scopes = FLAGS.excluded_scopes)

            quant_var = _get_quant_variable_from_model()

            # add quant var in dictionary for restore
            if isinstance(vars, dict):
                vars.update(quant_var)
                print("load quant var:\n\t" + "\n\t".join(["{}:{}".format(k, v) for k, v in vars.items()]))

            elif isinstance(vars, list):
                vars.extend(list(quant_var.values()))
                print("load quant var:\n\t" + "\n\t".join(["{}:{}".format(v.name[:-2], v) for v in vars]))


        labels = tf.squeeze(labels)
        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'Recall_5': slim.metrics.streaming_recall_at_k(
                logits, labels, 5),
        })

        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        ################
        # do evaluation#
        ################

    # initialize all variable
    with tf.Session(graph=g) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print_variables(vars)

        tf.logging.info("==========start test origin==========")

        if not FLAGS.wait_for_checkpoints:
            print("Do evaluation once.")

            if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
                tf.logging.info('latest checkpoint is %s' % checkpoint_path)
            else:
                checkpoint_path = FLAGS.checkpoint_path
            # print('trainable_var:\n\t%s' % "\n\t".join([ele.name for ele in tf.trainable_variables()]))
            # print('all_var:\n\t%s' % "\n\t".join([ele.name for ele in tf.all_variables()]))
            tf.logging.info('Evaluating %s' % checkpoint_path)

            assign_fn = slim.assign_from_checkpoint_fn(
                checkpoint_path,
                vars,
                ignore_missing_vars=FLAGS.ignore_missing_vars)

            assign_fn(sess)
            tf.summary.FileWriter(graph=tf.get_default_graph(),logdir=FLAGS.eval_dir)
            # saver.restore(sess, checkpoint_path)

            total_num = 0
            right_num = 0
            # validation loop
            for _ in tqdm.tqdm(range(num_batches)):
            # for _ in range(num_batches):
                try:
                    # get target from dataset
                    # img_np, label_np = sess.run([images, labels])
                    # get prediction
                    predictions_np,labels_np = sess.run([predictions,labels])

                except tf.errors.OutOfRangeError:
                    print("End of dataset.")
                    break
                #
                print("p:{}".format(predictions_np))
                print("l:{}".format(labels_np))

                labels_np = np.squeeze(labels_np)
                predictions_np = np.squeeze(predictions_np)
                total_num = total_num + len(predictions_np)
                match = np.equal(predictions_np, labels_np)
                batch_right_num = np.sum(match)
                right_num = right_num + batch_right_num

                # print("accu:{}".format(batch_right_num / len(predictions_np)))
            tf.logging.info("total sample:{}".format(total_num))
            tf.logging.info("right sample:{}".format(right_num))

            tf.logging.info("origin test accu:{}".format(right_num / total_num))

        else:

            print("Do evaluation by new checkpoint.")
            # Waiting loop.
            checkpoint_path = FLAGS.checkpoint_path
            tf.logging.info('Evaluating %s' % checkpoint_path)
            slim.evaluation.evaluation_loop(
                master = "",
                session_config=config,
                checkpoint_dir=checkpoint_path,
                logdir=FLAGS.eval_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                variables_to_restore=vars,
                eval_interval_secs=60,
                max_number_of_evaluations=np.inf,
                timeout=None,)



if __name__ == '__main__':
    tf.app.run()
