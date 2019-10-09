import sys
sys.path.append("../")
from pyRbRuntime import Network
import numpy as np
from cv_tools import preprocess

import Rb_Runtime_config

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def Rb_initial_net(proto, coeff):
    print("SGIR path:{}".format(proto))
    print("COEFF path:{}".format(coeff))
    net = Network(proto, coeff)
    return net

def Rb_infer(net,outputs,img):
    return net.Run(img,outputs)


def run():
    coeff_dir = Rb_Runtime_config.COEFF_DIR
    proto_path = Rb_Runtime_config.PROTOTXT_PATH
    image_dir = Rb_Runtime_config.INFER_IMG_DIR
    # the outcome list can check the graph by sg_go view model.json
    output_node_list = Rb_Runtime_config.OUTPUT_NODE_LIST

    net = Rb_initial_net(proto_path,coeff_dir)
    for file_name, image in preprocess.generate_img_inception(image_dir):
        # img, ratio_h, ratio_w =  preprocess.resize_img(image,Rb_Runtime_config.IMG_SHAPE[1],Rb_Runtime_config.IMG_SHAPE[0])
        # img_batch = np.expand_dims(image,0)
        # result is a dictionary saveing the numpy array
        result = Rb_infer(net=net,img = image, outputs=output_node_list)
        print("{} result:\n{}".format(file_name,np.argmax(result[output_node_list[0]])))
run()

