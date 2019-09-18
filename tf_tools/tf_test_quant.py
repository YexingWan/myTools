import sys,os
sys.path.append("/my_host/mnt/newhome/yexing/workspace/myTools/models_tensorflow/models/research/slim")
sys.path.append("..")
sys.path.append(".")
import tensorflow as tf
from keras import backend as K
from preprocessing import preprocessing_factory
import pathlib
import models_tensorflow.models.research.slim.nets.mobilenet_v1 as mobilenet_v1
import models_tensorflow.models.research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2
from tf_record_tools import load_record_classification_ILSVRC_val_dataset
import numpy as np
from tqdm import tqdm
import logging
import keras
from yolo import YOLO

"""
preprocessing_fn_map = {
      'cifarnet': cifarnet_preprocessing,
      'inception': inception_preprocessing,
      'inception_v1': inception_preprocessing,
      'inception_v2': inception_preprocessing,
      'inception_v3': inception_preprocessing,
      'inception_v4': inception_preprocessing,
      'inception_resnet_v2': inception_preprocessing,
      'lenet': lenet_preprocessing,
      'mobilenet_v1': inception_preprocessing,
      'mobilenet_v2': inception_preprocessing,
      'mobilenet_v2_035': inception_preprocessing,
      'mobilenet_v2_140': inception_preprocessing,
      'nasnet_mobile': inception_preprocessing,
      'nasnet_large': inception_preprocessing,
      'pnasnet_mobile': inception_preprocessing,
      'pnasnet_large': inception_preprocessing,
      'resnet_v1_50': vgg_preprocessing,
      'resnet_v1_101': vgg_preprocessing,
      'resnet_v1_152': vgg_preprocessing,
      'resnet_v1_200': vgg_preprocessing,
      'resnet_v2_50': vgg_preprocessing,
      'resnet_v2_101': vgg_preprocessing,
      'resnet_v2_152': vgg_preprocessing,
      'resnet_v2_200': vgg_preprocessing,
      'vgg': vgg_preprocessing,
      'vgg_a': vgg_preprocessing,
      'vgg_16': vgg_preprocessing,
      'vgg_19': vgg_preprocessing,
  }"""



mobilenet_checkpoint_path = "/my_host/mnt/newhome/yexing/checkpoint/mobilenet_v1/mobilenet_v1_1.0_224.ckpt"
yolo_checkpoint_path = ""
val_record_dataset = "/my_host/mnt/newhome/mengyong/datasets/imagenet"
preprocessing_name = "mobilenet_v1"
logging.basicConfig(level=logging.INFO)



def get_imagenet_val_data(val_record_dir,
             h,
             w,
             is_training,
             preprocessing_name,
             batch_size):
    # get dataset and iteration for validation
    dataset = load_record_classification_ILSVRC_val_dataset(val_record_dir)
    iterator = dataset.make_one_shot_iterator()
    img, label = iterator.get_next()
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=is_training)
    img = image_preprocessing_fn(img, h, w)
    imgs, labels = tf.train.batch(
        [img, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=10* batch_size)

    return imgs, labels, dataset, iterator


# TODO: not finish: calculate mAP
def quant_yolo(checkpoint_path:str,
               val_record_dir:str,
               anchors_path:str,
               classes_path:str,
               gpu_num:int = 1,
               input_size: tuple = (416, 416),
               score:float = 0.3,
               iou:float = 0.2,
               number_img: int= 10000,
               batch_size: int = 32):

    iter_num = int(number_img / batch_size)
    yolo_config = {
        "model_path": checkpoint_path,
        "anchors_path": anchors_path,
        "classes_path": classes_path,
        "score" : score,
        "iou" : iou,
        "model_image_size" : input_size,
        "gpu_num" : gpu_num,
    }
    # initial model
    yolo = YOLO(**yolo_config)
    sess = tf.Session()

    # get dataset and iteration for validation
    imgs, labels,dataset, iterator = get_imagenet_val_data(val_record_dir,input_size[0],input_size[1],False,'resnet_v1_50',batch_size)

    total_num = 0
    right_num = 0

    logging.info("==========start test origin==========")
    # validation loop
    for _ in tqdm(range(iter_num)):
        try:
            # get target from dataset
            img_np, label_np = sess.run([imgs, labels])
            # get prediction
            boxes,scores,classes = yolo.sess.run([yolo.boxes,yolo.scores,yolo.classes],
                                            feed_dict={
                                                yolo.yolo_model.input: img_np,
                                                yolo.input_image_shape: [img_np.size[1], img_np.size[0]],
                                                K.learning_phase(): 0
                                            })
        except tf.errors.OutOfRangeError:
            print("End of dataset.")
            break


        # total_num = total_num + batch_size
        # match = np.equal(class_infer_ori, label_np)
        # batch_right_num = np.sum(match)
        # right_num = right_num + batch_right_num

    logging.info("origin test accu:{}".format(right_num / total_num))

        # ========================================== tf-lite quant =======================================

    total_num = 0
    right_num = 0

    # use tf.lite to quant model to 8-bits and save
    converter = tf.lite.TFLiteConverter.from_keras_model_file(checkpoint_path)
    converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                            tf.lite.OpsSet.SELECT_TF_OPS]

    # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    def representative_dataset_gen():
        print("in")
        iterator = dataset.make_one_shot_iterator()
        print("iterator")
        img, label = iterator.get_next()
        for _ in tqdm(range(1500), "calibrating"):
            img_np, _ = sess.run([img, label])
            logging.debug("feed data shape:{}".format(img_np.shape))
            # Get sample input data as a numpy array in a method of your choosing.
            yield [img_np]

    # set converter
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.allow_custom_ops = True

    # do quantization and save tflite file
    tflite_model = converter.convert()

    tflite_models_dir = pathlib.Path("/tmp/tf_lite_model/")
    tflite_models_dir.mkdir(exist_ok=True, parents=True)
    tflite_model_file = tflite_models_dir / "tf_lite_model.tflite"
    tflite_model_file.write_bytes(tflite_model)

    # create interpreter for prediction
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    imgs, labels,dataset, iterator = get_imagenet_val_data(val_record_dir,input_size[0],input_size[1],False,'resnet_v1_50',batch_size)

    logging.info("==========start test quant==========")

    for _ in tqdm(range(iter_num)):
        try:
            img_np, label_np = sess.run([imgs, labels])
        except tf.errors.OutOfRangeError:
            print("End of dataset.")
            break

        # do prediction using interpreter
        interpreter.set_tensor(input_index, img_np)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)
        total_num = total_num + batch_size
        match = np.equal(predictions, label_np)
        batch_right_num = np.sum(match)
        right_num = right_num + batch_right_num

    logging.info("quant test accu:{}".format(right_num / total_num))




    pass




def quant_mobileNet(checkpoint_path,
                    val_record_dir,
                    number_img,
                    batch_size,
                    input_size = (224,224),
                    network = "v1"):
    """

    :param checkpoint_path:
    :param val_record_dir:
    :param number_img:
    :param batch_size:
    :param input_size:
    :param network: "v1"/"v2"
    :return:
    """

    total_num = 0
    right_num = 0
    iter_num = int(number_img / batch_size)
    sess = tf.Session()
    preprocessing_name = "mobilenet_v1" if network == "v1" else "mobilenet_v2"
    with sess.as_default():
        sess = tf.get_default_session()

        imgs, labels, dataset, iterator = get_imagenet_val_data(val_record_dir, input_size[0], input_size[1], False,
                                                                preprocessing_name, batch_size)

        ph_img = tf.placeholder(tf.float32,(batch_size,*input_size),"input")
        if network == "v1":
            with tf.contrib.slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)):
                net_v1, endpoints = mobilenet_v1.mobilenet_v1(ph_img,num_classes=1001)
                prediction = endpoints["Predictions"]
                prediction = tf.argmax(prediction, axis=1)
                graph = tf.get_default_graph()
                tf.summary.FileWriter(logdir="./graph_mobilev1",graph=graph)
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_path)

        elif network == "v2":
            with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
                net_v2, _ = mobilenet_v2.mobilenet(ph_img)
                # get predict index Tensor as output
                prediction = tf.argmax(net_v2,axis=1)
                prediction = tf.argmax(prediction, axis=1)
                graph = tf.get_default_graph()
                tf.summary.FileWriter(logdir="./graph_mobilev2", graph=graph)
                # load moving average checkpoint (used in mobilev2)
                ema = tf.train.ExponentialMovingAverage(0.999)
                vars = ema.variables_to_restore()
                saver = tf.train.Saver(vars)
                saver.restore(sess, checkpoint_path)

        logging.info("==========start test origin==========")
        # validation loop
        for _ in tqdm(range(iter_num)):
            try:
                # get target from dataset
                img_np, label_np = sess.run([imgs, labels])
                # get prediction
                class_infer_ori= sess.run([prediction],feed_dict={ph_img:img_np})


            except tf.errors.OutOfRangeError:
                print("End of dataset.")
                break
            total_num = total_num + batch_size
            match = np.equal(class_infer_ori,label_np)
            batch_right_num = np.sum(match)
            right_num = right_num +  batch_right_num

        logging.info("origin test accu:{}".format(right_num/total_num))


        #========================================== tf-lite quant =======================================

        total_num = 0
        right_num = 0

        # use tf.lite to quant model to 8-bits and save
        converter = tf.lite.TFLiteConverter.from_session(sess,[ph_img],[prediction])
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.SELECT_TF_OPS]

        # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        def representative_dataset_gen():
            print("in")
            iterator = dataset.make_one_shot_iterator()
            print("iterator")
            img, label = iterator.get_next()
            for _ in tqdm(range(1500),"calibrating"):
                img_np, _ = sess.run([img, label])
                logging.debug("feed data shape:{}".format(img_np.shape))
                # Get sample input data as a numpy array in a method of your choosing.
                yield [img_np]
        # set converter
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.allow_custom_ops = True

        # do quantization and save tflite file
        tflite_model = converter.convert()

        tflite_models_dir = pathlib.Path("/tmp/tf_lite_model/")
        tflite_models_dir.mkdir(exist_ok=True, parents=True)
        tflite_model_file = tflite_models_dir / "tf_lite_model.tflite"
        tflite_model_file.write_bytes(tflite_model)

        # create interpreter for prediction
        interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]


        imgs, labels, dataset, iterator = get_imagenet_val_data(val_record_dir, input_size[0], input_size[1], False,
                                                                preprocessing_name, batch_size)
        logging.info("==========start test quant==========")

        for _ in tqdm(range(iter_num)):
            try:
                img_np, label_np = sess.run([imgs,labels])
            except tf.errors.OutOfRangeError:
                print("End of dataset.")
                break

            # do prediction using interpreter
            interpreter.set_tensor(input_index, img_np)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)
            total_num = total_num + batch_size
            match = np.equal(predictions,label_np)
            batch_right_num = np.sum(match)
            right_num = right_num +  batch_right_num

        logging.info("quant test accu:{}".format(right_num/total_num))














