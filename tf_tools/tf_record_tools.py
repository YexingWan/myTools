import glob
import io
import os

import PIL
import numpy as np
import tensorflow as tf
from PIL import Image


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_record_classification_ILSVRC_val_dataset(record_dir: str,
                                                  file_search_name="*val*.record"):
    """
    :param record_dir: path to all record file
    :param batch_size: batch size
    :param input_size: HWC
    :return:
    """

    def record_parse_function(serial_exmp):
        keys_to_features = {
            'name': tf.FixedLenFeature([], tf.string, default_value=''),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature((), tf.string, default_value=''),
            'label': tf.FixedLenFeature([], tf.int64),
            'labeltext': tf.FixedLenFeature([], tf.string, default_value=''),
            'format': tf.FixedLenFeature([], tf.string, default_value='raw')
        }

        feats = tf.parse_single_example(serial_exmp, features=keys_to_features)
        image = tf.decode_raw(feats['image'], tf.uint8)
        image = tf.reshape(image, tf.stack([feats['height'], feats['width'], 3]))
        feats.update({'image': image})
        return feats["image"], feats["label"]

    # filenames = glob.glob(os.path.join(record_dir, file_search_name))
    # dataset = tf.data.TFRecordDataset(filenames)
    # assert (len(filenames) != 0, "No record file found. Please check the path.")

    # do Parallelize data extraction
    files = tf.data.Dataset.list_files(os.path.join(record_dir, file_search_name))
    dataset = files.interleave(
        tf.data.TFRecordDataset, cycle_length=3,
        num_parallel_calls=3)
    dataset = dataset.map(record_parse_function,num_parallel_calls=3)
    dataset = dataset.shuffle(buffer_size=500)

    return dataset


def save_record_classification_ILSVRC_val_dataset(image_dir: str,
                                                  annotation_file_name,
                                                  label_file_name,
                                                  record_dir,
                                                  tag="val",
                                                  number_sample_pre_file=1000):
    assert (os.path.exists(image_dir), "Image directory {} not exist.".format(image_dir))
    annotation_file = os.path.join(image_dir, annotation_file_name)
    assert (os.path.exists(annotation_file), "Anotation file {} not exist.".format(annotation_file))
    label_file = os.path.join(image_dir, label_file_name)
    assert (os.path.exists(label_file), "Label file {} not exist.".format(label_file))


    # load label file and construct dict
    numer_sample = 0
    label_file = open(label_file, "r")
    label_text_dict = {}
    for line in label_file:
        sample = line.split(" ")
        label_text_dict[int(sample[0])] = sample[1]
    label_file.close()

    # read annotation file and write record
    writer = tf.python_io.TFRecordWriter(record_dir, "data.tfrecords.%s.-%.5d-to-%.5d.record" % (
    tag, 0, number_sample_pre_file - 1))
    annotation_file = open(annotation_file, "r")
    for idx, line in enumerate(annotation_file):
        sample = line.split(" ")
        img_name = sample[0]
        img_path = os.path.join(image_dir, sample[0])
        img = Image.open(img_path)
        img_raw = img.tobytes()
        label = int(sample[1])
        label_text = label_text_dict[label]

        feature_internal = {
            'name': _bytes_feature(img_name),
            'height': _int64_feature(img.shape[1]),
            'width': _int64_feature(img.shape[0]),
            'image': _bytes_feature(img_raw),
            'label': _int64_feature(label),
            'labeltext': _bytes_feature(label_text),
            'format': _bytes_feature("raw")
        }

        features = tf.train.Features(feature_internal)
        example = tf.train.Example(features)
        example_raw = example.SerializeToString()
        writer.write(example_raw)
        numer_sample += 1
        # update writer
        if idx + 1 % number_sample_pre_file == 0:
            record_path = os.path.join(record_dir, "data.tfrecords.%s.-%.5d-to-%.5d.record" % (
            tag, idx + 1, idx + number_sample_pre_file - 1))
            writer = tf.python_io.TFRecordWriter(record_path)
        f = open(os.path.join(record_dir,"num_sample"),"w")
        f.write(numer_sample)
        f.close()

def load_record_classification_ILSVRC_train_dataset(record_dir: str,
                                                    file_search_name="*train*.record"):
    return load_record_classification_ILSVRC_val_dataset(record_dir, file_search_name)


def save_record_classification_train_dataset(image_dir: str,
                                             annotation_file_name,
                                             label_file_name,
                                             record_dir,
                                             tag="train",
                                             number_sample_pre_file=1000):
    save_record_classification_ILSVRC_val_dataset(image_dir,
                                                         annotation_file_name,
                                                         label_file_name,
                                                         record_dir,
                                                         tag,
                                                         number_sample_pre_file)


def load_record_classification_ILSVRC_infer_dataset(record_dir: str,
                                                  file_search_name="*infer*.record") -> tf.data.Dataset:

    def record_parse_function(serial_exmp):
        keys_to_features = {
            'name': tf.FixedLenFeature([], tf.string, default_value=''),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature((), tf.string, default_value=''),
            'format': tf.FixedLenFeature([], tf.string, default_value='raw')
        }

        feats = tf.parse_single_example(serial_exmp, features=keys_to_features)
        image = tf.decode_raw(feats['image'], tf.uint8)
        image = tf.reshape(image, tf.stack([feats['height'], feats['width'], 3]))
        feats.update({'image': image})
        return feats["image"]

    filenames = glob.glob(os.path.join(record_dir, file_search_name))
    assert (len(filenames) != 0, "No record file found. Please check the path.")
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(record_parse_function)

    return dataset


def save_record_classification_ILSVRC_infer_dataset(image_dir: str,
                                             record_dir,
                                             tag="infer",
                                             number_sample_pre_file=1000):
    assert (os.path.exists(image_dir), "Image directory {} not exist.".format(image_dir))

    # load label file and construct dict
    numer_sample = 0

    # read annotation file and write record
    writer = tf.python_io.TFRecordWriter(record_dir, "data.tfrecords.%s.-%.5d-to-%.5d.record" % (
        tag, 0, number_sample_pre_file - 1))

    for idx, img_path in enumerate(glob.glob(image_dir)):
        img_name = os.path.split(img_path)[-1]
        img = Image.open(img_path)
        img_raw = img.tobytes()

        feature_internal = {
            'name': _bytes_feature(img_name),
            'height': _int64_feature(img.shape[1]),
            'width': _int64_feature(img.shape[0]),
            'image': _bytes_feature(img_raw),
            'format': _bytes_feature("raw")
        }

        features = tf.train.Features(feature_internal)
        example = tf.train.Example(features)
        example_raw = example.SerializeToString()
        writer.write(example_raw)
        numer_sample += 1
        # update writer
        if idx + 1 % number_sample_pre_file == 0:
            record_path = os.path.join(record_dir, "data.tfrecords.%s.-%.5d-to-%.5d.record" % (
                tag, idx + 1, idx + number_sample_pre_file - 1))
            writer = tf.python_io.TFRecordWriter(record_path)
        f = open(os.path.join(record_dir, "num_sample"), "w")
        f.write(numer_sample)
        f.close()



def load_record_detection_coco_train_dataset(record_dir: str,
                                             file_search_name="*val*.record"):

    """
    record feature dic:
    feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))



    :param record_dir: path to all record file
    :param batch_size: batch size
    :param input_size: HWC
    :return:
    """

    def record_parse_function(serial_exmp):
        keys_to_features = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/source_id': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/key/sha256': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/object/bbox/xmin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.VarLenFeature(tf.float32),
            'image/object/class/text':
                tf.VarLenFeature(tf.string),
            'image/object/is_crowd':
                tf.VarLenFeature(tf.int64),
            'image/object/area':
                tf.VarLenFeature(tf.int64),
        }

        feats = tf.parse_single_example(serial_exmp, features=keys_to_features)
        encoded_jpg = feats['image/encoded']
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)
        feats.update({'image': image})

        return feats["image"],feats["image/object/bbox/xmin"],feats['image/object/bbox/xmax'],\
               feats['image/object/bbox/ymin'],feats['image/object/bbox/ymax'],feats['image/object/class/text']

    filenames = glob.glob(os.path.join(record_dir, file_search_name))
    assert (len(filenames) != 0, "No record file found. Please check the path.")
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(record_parse_function)

    return dataset


def save_record_detection_coco_train_dataset(record_dir: str,
                                             batch_size: int,
                                             input_size: tuple = None,
                                             file_search_name="*val*.record"):











    """
    record feature dic:
    feature_dict = {
      'image/height':
          dataset_util.int64_feature(image_height),
      'image/width':
          dataset_util.int64_feature(image_width),
      'image/filename':
          dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id':
          dataset_util.bytes_feature(str(image_id).encode('utf8')),
      'image/key/sha256':
          dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded':
          dataset_util.bytes_feature(encoded_jpg),
      'image/format':
          dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin':
          dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax':
          dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin':
          dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax':
          dataset_util.float_list_feature(ymax),
      'image/object/class/text':
          dataset_util.bytes_list_feature(category_names),
      'image/object/is_crowd':
          dataset_util.int64_list_feature(is_crowd),
      'image/object/area':
          dataset_util.float_list_feature(area),
  }
  if include_masks:
    feature_dict['image/object/mask'] = (
        dataset_util.bytes_list_feature(encoded_mask_png))



    :param record_dir: path to all record file
    :param batch_size: batch size
    :param input_size: HWC
    :return:
    """

    def record_parse_function(serial_exmp):
        keys_to_features = {
            'image/height': tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/source_id': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/key/sha256': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/format': tf.FixedLenFeature([], tf.string, default_value=''),
            'image/object/bbox/xmin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin':
                tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax':
                tf.VarLenFeature(tf.float32),
            'image/object/class/text':
                tf.VarLenFeature(tf.string),
            'image/object/is_crowd':
                tf.VarLenFeature(tf.int64),
            'image/object/area':
                tf.VarLenFeature(tf.int64),
        }

        feats = tf.parse_single_example(serial_exmp, features=keys_to_features)
        encoded_jpg = feats['image/encoded']
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = PIL.Image.open(encoded_jpg_io)

        # image = tf.decode_raw(feats['image/encoded'],)
        image = tf.reshape(image, tf.stack([feats['height'], feats['width'], 3]))
        image = tf.cast(image, tf.float32) / 128. - 1
        image = tf.expand_dims(image, 0)
        if input_size is not None:
            image = tf.image.resize_bilinear(image, np.array([input_size[0], input_size[1]]))
        image = tf.squeeze(image)

        feats.update({'image': image})

        return feats["image"], feats["label"]

    filenames = glob.glob(os.path.join(record_dir, file_search_name))
    assert (len(filenames) != 0, "No record file found. Please check the path.")
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(record_parse_function).batch(batch_size)

    return dataset

