import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os

ckpt_path = "./checkpoints/testmodel"
graph_def_string = "./checkpoints/input_graph_def.pd"
if not os.path.exists("./checkpoints"):
    os.mkdir("./checkpoints")


def init_and_save_ckpt(sess,ckpt_path):
    saver = tf.train.Saver()
    saver.save(sess,ckpt_path)

def print_node(g:tf.Graph):
    gd = g.as_graph_def()
    for node in gd.node:
        print(node)

def print_Conv_op(g: tf.Graph):
    for n in g.get_operations():
        if n.node_def.op == "Conv2D":
            print(n.node_def)

# try to match variable NodeDef to Conv2D NodeDef
def print_Conv_match_weight(g: tf.Graph):
    for n in g.get_operations():
        if n.node_def.op == "Conv2D":
            print(n.node_def.name+":")
            for i in n.node_def.input:
                if "weights" in i:
                    weight_read = g.get_operation_by_name(i)
                    for i_ in weight_read.node_def.input:
                        if "weights" in i_:
                            print(g.get_operation_by_name(i_))
            print()

def get_Conv_weigt_nodeDef(conv_nd: tf.NodeDef):
    assert(conv_nd.op == "Conv2D")
    print(conv_nd.name)
    for i in conv_nd.input:
        if "weights" in i:
            weight_read_op = g.get_operation_by_name(i)
            for i_ in weight_read_op.node_def.input:
                if "weights" in i_:
                    weight_op  = g.get_operation_by_name(i_)
                    print(weight_op.node_def.name)
                    return weight_op.node_def



g = tf.Graph()
with g.as_default():
    image = tf.placeholder(tf.float32, (1, 256, 256, 3), name="input")

    with tf.name_scope("subnet1"):
        l1 = slim.conv2d(image,64,3,normalizer_fn=slim.batch_norm)
        l2 = slim.conv2d(l1,64,3,normalizer_fn=slim.batch_norm)
        l3 = slim.conv2d(l2,20,3,normalizer_fn=slim.batch_norm)
        re = tf.reduce_mean(l3,axis=(0,1,2))
    with tf.name_scope("subnet2"):
        l1_ = slim.conv2d(image, 64, 3, normalizer_fn=slim.batch_norm)
        l2_ = slim.conv2d(l1_, 64, 3, normalizer_fn=slim.batch_norm)
        l3_ = slim.conv2d(l2_, 20, 3, normalizer_fn=slim.batch_norm)
        re_ = tf.reduce_mean(l3_, axis=(0, 1, 2))
    result = tf.add(re,re_)

# random initial all variable
with tf.Session(graph=g) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    rand_in = np.random.rand(1,256,256,3)
    re = sess.run(result,feed_dict={image:rand_in})
    for v in tf.global_variables():
        print(v.name)
    print("==================================")
    # gd = g.as_graph_def()
    # for node in gd.


    nd = get_Conv_weigt_nodeDef(g.get_operation_by_name("subnet2/Conv/Conv2D").node_def)
    nd.attr["shape"].shape.dim[0].size=512
    print(nd.attr["shape"].shape.dim[0].size)

    # print_Conv_op(g)
    # do experiment of subnet2/Conv/Conv2D















