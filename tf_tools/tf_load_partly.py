import tensorflow as tf

g = tf.Graph()
with g.as_default():
    with tf.Session(graph=g) as sess:
        tf.train.import_meta_graph("./tep/v1.ckpt.meta")
        new_saver = tf.train.Saver(tf.global_variables(scope='v1'))
        new_saver.restore(sess,"./tep/v1.ckpt")
        sess.run(tf.global_variables_initializer())
        print(tf.global_variables())
        writer = tf.summary.FileWriter("./tep/graph_vis",graph=g)
        writer.close()
