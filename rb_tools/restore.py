import tensorflow as tf
import os
import sys
sys.path.append("../")
import tensorflow.contrib.slim as slim
from tf_tools.tf_print import print_variables, print_checkpoint



def tf_basic_restore(model_constructor,
                     checkpoint_path:str,
                     restore_dir = "./restored",
                     input_shape:tuple = (224,224),
                     model_name="model",
                     from_ema=False,
                     ignore_missing_vars = False):
    g = tf.Graph()
    with g.as_default():
        # add placeholder
        preprocessed_image = tf.placeholder(dtype=tf.float32, shape=(1,*input_shape,3))
        # please make sure the model has no input layer (something like DataIterator or Provider)
        model_constructor(preprocessed_image)
        with tf.Session(graph=g) as sess:
            if os.path.isdir(checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
                tf.logging.info('latest checkpoint is %s' % checkpoint_path)
            print("Restore from %s" % checkpoint_path)

            # Deal with ema checkpoint
            if from_ema:
                variable_averages = tf.train.ExponentialMovingAverage(0.999)
                vars = variable_averages.variables_to_restore(slim.get_model_variables())
            else:
                # ema.variables_to_restore will generate proper dictionary base one global_variables
                # so we need to use global_varibales here.
                vars = tf.global_variables()

            print_variables(vars)
            print_checkpoint(checkpoint_path)

            # load checkpoint
            assign_fn = slim.assign_from_checkpoint_fn(
                checkpoint_path,
                vars,
                ignore_missing_vars=ignore_missing_vars)
            assign_fn(sess)




            if not os.path.exists(restore_dir):
                os.mkdir(restore_dir)
            if not os.path.exists(os.path.join(restore_dir,"graph_vis")):
                os.mkdir(os.path.join(restore_dir,"graph_vis"))
            saver = tf.train.Saver(vars)
            saver.save(sess,os.path.join(restore_dir,model_name)+'.ckpt')

            writer = tf.summary.FileWriter(os.path.join(restore_dir,"graph_vis"), sess.graph)
            writer.close()