import numpy as np
import cv2, os
import tensorflow as tf
from pyRbRuntime import Network
from collections import OrderedDict


def Rb_initial_net(proto, coeff):
    print("SGIR path:{}".format(proto))
    print("COEFF path:{}".format(coeff))
    net = Network(proto, coeff)
    return net

def Rb_infer(net,outputs,img):
    return net.Run(img,outputs)


def tf_basic_restore(model_constructor,checkpoint_path:str,inputshape:tuple):
    g = tf.Graph()
    with g.as_default():

        preprocessed_image = tf.placeholder(dtype=tf.float32, shape=inputshape)
        with tf.Session(graph=g) as sess:
            saver = tf.train.Saver()
            model_constructor(preprocessed_image)
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print("Restore from %s" % model_path)
            saver.restore(sess, model_path)

            if not os.path.exists("./restored"):
                os.mkdir("./restored")
            if not os.path.exists("./restored/graph_vis"):
                os.mkdir('./restored/graph_vis')

            saver.save(sess, "./restored/EAST.ckpt")
            writer = tf.summary.FileWriter("./restored/graph_vis", sess.graph)
            writer.close()

def pytorch_load_state_dict_no_parallel(model, state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def pytorch_flatten_print_model(model):
    for m_name,m in model.named_modules():
        if not list(m.children()):
            print("[", m_name, "] -> ",m)
            print("[parameters] -> ")
            for idx, (p_name, p) in enumerate(m.named_parameters()):
                print(idx, "-> [", p_name, "] :", p.size())
            print("="*100)


# Gamma correction to input image
def gamma_transfer(img):
    gamma = 0.4
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return res


def resize_img(img, n_h, n_w):
    h, w, _ = img.shape
    ratio_h = n_h / float(h)
    ratio_w = n_w / float(w)
    img = cv2.resize(img, (n_w, n_h), interpolation=cv2.INTER_LINEAR)
    return img, ratio_h, ratio_w


def general_generate_img(image_dir, new_height, new_width, mean_sub=True):
    mean_array = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    # image_dir = glob(os.path.join(os.path.abspath(image_dir),"*.jpg"))
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))

    for p in files:
        img_ori = cv2.imread(p)  # BGR
        # img = image_preprocess(img_ori)
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)

        img, ratio_h, ratio_w = resize_img(img, new_height, new_width)
        print("ratio: %.5f, %.5f" % (ratio_h, ratio_w))
        if mean_sub:
            img = img - mean_array

        img_batch = np.expand_dims(img, 0)
        yield ratio_w, ratio_h, p, img_ori, img_batch


def sigmoid(x):
    return 1. / (1. + np.exp(-x))



def tool_demo():
    coeff_dir = ''
    proto_dir = ''
    image_dir = "dir/to/image"
    # the outcome list can check the graph by sg_go view model.json
    outcomes_list = [
        'feature_fusion/Conv_8/Conv2D',
        'feature_fusion/Conv_9/Conv2D',
        'feature_fusion/Conv_7/Conv2D',
    ]
    net = Rb_initial_net(proto_dir,coeff_dir)
    for ratio_w, ratio_h, p, img_ori, img_batch in general_generate_img(image_dir,224,224):
        # result is a dictionary saveing the numpy array
        result = Rb_infer(img_batch,outcomes_list)
        geo_f = result['feature_fusion/Conv_8/Conv2D']
        angle_f = result['feature_fusion/Conv_9/Conv2D']
        score = result['feature_fusion/Conv_7/Conv2D']





