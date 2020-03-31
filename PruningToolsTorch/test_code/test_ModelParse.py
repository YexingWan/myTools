import torch,torchvision
import torch.nn as nn
import numpy as np
import logging
import sys
# sys.path.append("/mnt/newhome/yexing/workspace/PyTorch-YOLOv3")
# from models import Darknet
import tensorboardX
from torchvision import models
import re


# deep_wise conv config: groups != 1 or node.inputs()[9] != 1 (how to get node.inputs()[9]?)
# at::_convolution(input, weight, bias, stride, padding, dilation,
#                         transposed, output_padding, groups,
#                         ctx.benchmarkCuDNN(), ctx.deterministicCuDNN(), ctx.userEnabledCuDNN());


# model_sample = torchvision.models.resnet50()



# Initiate Darknet
# weights_path = "/home/yx-wan/newhome/workspace/PyTorch-YOLOv3/weights/yolov3.weights"
# model_def = "/home/yx-wan/newhome/workspace/PyTorch-YOLOv3/config/yolov3.cfg"
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_sample = Darknet(model_def).to("cpu")
# if weights_path.endswith(".weights"):
#     # Load darknet weights
#     model_sample.load_darknet_weights(weights_path)
# else:
#     # Load checkpoint weights
#     model_sample.load_state_dict(torch.load(weights_path))







class Net_with_ModuleList_and_Squence(torch.nn.Module):
    def __init__(self):
        super(Net_with_ModuleList_and_Squence, self).__init__()
        # self.module_list = build_model()

        self.bl1 = nn.ModuleList()
        self.bl1.append(nn.Conv2d(3, 32, 3, padding=1, bias=False))
        self.bl1.append(nn.Conv2d(32, 31, 3, padding=1, bias=False))
        self.bl1.append(nn.Conv2d(31, 30, 3, padding=1, bias=False))




        self.bl2 = nn.ModuleList()
        self.bl2.append(nn.Conv2d(30, 29, 3, padding=1, bias=False))
        self.bl2.append(nn.Conv2d(29, 29, 3, padding=1, bias=False))
        self.bl2.append(nn.Conv2d(29, 28, 3, padding=1, bias=False))


        self.Sequential_1 = nn.Sequential()
        self.Sequential_1.add_module("conv",nn.Conv2d(28, 28, 3, padding=1, bias=False))
        self.Sequential_1.add_module("conv",nn.Conv2d(28, 28, 3, padding=1, bias=False))
        self.Sequential_1.add_module("conv",nn.Conv2d(28, 28, 3, padding=1, bias=False))



        self.Sequential_2 = nn.Sequential(nn.ReLU(),
                                          nn.Conv2d(28, 28, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(28))
        self.Sequential_3 = nn.Sequential(nn.ReLU(),
                                          nn.Conv2d(28, 28, 3, padding=1, bias=False),
                                          nn.BatchNorm2d(28))






    def forward(self, x):
        for m in self.bl1:
            x = m(x)
        for m in self.bl2:
            x = m(x)
        x = self.Sequential_1(x)
        x = self.Sequential_2(x)
        x = self.Sequential_3(x)

        return x

# model_sample = Net_with_ModuleList_and_Squence()
# model_sample.bl1_conv0 = model_sample.bl1[0]
# model_sample.bl1_conv1 = model_sample.bl1[1]
# model_sample.bl1_conv2 = model_sample.bl1[2]
#
# model_sample.bl2_conv0 = model_sample.bl2[0]
# model_sample.bl2_conv1 = model_sample.bl2[1]
# model_sample.bl2_conv2 = model_sample.bl2[2]

model_sample = models.resnet50()                               # success

print("========Graph Tree========")
for name, m in model_sample.named_modules():
    print("{}:{} => {}".format(name,type(m),m))


model_sample.eval()
# input_sample = torch.Tensor(np.random.randn(1,3,224,224))
input_sample = torch.Tensor(np.random.randn(1,3,224,224))
writer = tensorboardX.SummaryWriter(logdir="./graph")
writer.add_graph(model_sample,input_sample)
writer.close()
model_trace = torch.jit.trace(model_sample,input_sample,check_trace=False)
model_graph = model_trace.graph


def test_jit():
    print("========All tail Sub-module========")
    for name, m in model_sample.named_modules():
        if not list(m.children()):
            print("{}\t\t=> {}".format(name, m))



    print("\n\n========Model Graph========")
    print(model_graph)

    print_list_node(model_graph.nodes(),"All Graph Nodes")


    convolution_node = []
    for node in model_graph.nodes():
        if "convolution" in node.kind():
            # print(["{}({},{}):{}".format(n.uniqueName(),n.type().sizes(),node.kind(),n.node()) for n in list(node.outputs())])
            output_value = list(node.outputs())[0]
            if len(output_value.type().sizes()) == 4:
                convolution_node.append(node)
    print_list_node(convolution_node,"All Convolution Node")


    input_node_list = model_graph.inputs()
    print("========{}========".format("ALL input value"))
    print(list(input_node_list))


    print("========Attribute of torch._C.Graph without ‘_’========")
    print("{}\t\t=> {}\n".format(type(model_graph), [d for d in model_graph.__dir__() if "_" not in d]))

    # node = convolution_node[0]
    # print("========Attribute of torch._C.Node without ‘_’========")
    # print("{}\t\t=> {}\n".format(type(node), [d for d in node.__dir__() if '_' not in d]))
    node = convolution_node[0]
    print("========Attribute of torch._C.Node========")
    print("{}\t\t=> {}\n".format(type(node), [d for d in node.__dir__()]))

    value = list(node.inputs())[0] if list(node.inputs()) else list(node.outputs())[0]
    print("========Attribute of torch._C.Value without ‘_’========")
    print("{}\t\t=> {}\n".format(type(value), [d for d in value.__dir__() if "_" not in d]))


    print("========Attribute of torch._C.Type without ‘_’========")
    print("{}\t\t=> {}\n".format(type(value.type()), [d for d in value.type().__dir__() if "_" not in d]))


    print("========Attribute of torch._C.USE without ‘_’========")
    print("{}\t\t=> {}\n".format(type(value.uses()[0]), [d for d in value.uses()[0].__dir__() if "_" not in d]))


def print_list_node(nodes,title = "Nodes"):
    print("========{}========".format(title))
    print("\n".join(["{}\t\t=> {}:{}".format(scope2name(n.scopeName()),n,n.kind()) for n in nodes]))

def scope2name(scope):
    patern = r"(?<=\[)\w+(?=\])"
    return '.'.join(re.findall(patern,scope))




logging.basicConfig(level = logging.INFO)
print(model_sample)
test_jit()
