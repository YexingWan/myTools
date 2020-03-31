from tensorflow.python.platform import gfile 
import tensorflow as tf


graph = tf.get_default_graph()
graphdef = graph.as_graph_def()

_ = tf.train.import_meta_graph("2000/model-2000.ckpt.meta")
summary_write = tf.summary.FileWriter("./log" , graph)




