import tensorflow as tf
#os.environ['CUDA_VISIBLE_DEVICES']=3
g = tf.Graph()

with g.as_default():
    with tf.Session(graph=g) as sess:
        a = tf.Variable([1,2,3,4,5],name='v1',dtype=tf.float32)
        b = tf.Variable([2,3,4,5,6],name='v2',dtype=tf.float32)
        c = b-a
        # Creates a session with log_device_placement set to True.
        sess.run(tf.global_variables_initializer())
        variables = tf.global_variables(scope='v1')
        print(variables)
        tf.train.Saver(variables).save(sess,"./tep/v1.ckpt")





