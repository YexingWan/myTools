import os
import sys
import tensorflow as tf
from object_detection.utils import config_util
import functools
from object_detection.builders import model_builder
sys.path.append("../")
sys.path.append("../../models_tensorflow/models/research/slim")
from restore import tf_basic_restore

tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags

flags.DEFINE_string(
    'checkpoint_path', '',
    'Directory containing checkpoints to evaluate, typically '
    'set to `train_dir` used in the training job.')
flags.DEFINE_string('restore_path', './', 'Directory to write restore_path summaries to.')
flags.DEFINE_string('input_shape', '640,640', 'input size of model, "h,w".')
flags.DEFINE_string('model_name', 'model', 'names pb of files saved')
flags.DEFINE_boolean('ignore_missing_vars',False,"as the name mean")

flags.DEFINE_string(
    'pipeline_config_path', '',
    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
    'file. If provided, other configs are ignored')

FLAGS = flags.FLAGS



def main():
    tf.gfile.MakeDirs(FLAGS.restore_path)
    if FLAGS.pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(
            FLAGS.pipeline_config_path)
        tf.gfile.Copy(
            FLAGS.pipeline_config_path,
            os.path.join(FLAGS.restore_path, 'pipeline.config'),
            overwrite=True)
    else:
        raise RuntimeError('pipline config file must be provided')

    model_config = configs['model']
    model_fn = functools.partial(
        model_builder.build, model_config=model_config, is_training=False)
    input_shape = tuple([int(e) for e in str(FLAGS.input_shape).split(',')])


    tf_basic_restore(model_fn,
                     model_name=FLAGS.model_name,
                     checkpoint_path=FLAGS.checkpoint_path,
                     input_shape=input_shape,
                     ignore_missing_vars=FLAGS.ignore_missing_vars,
                     restore_dir=FLAGS.restore_path)


if __name__ == "__main__":
    main()





