import sys
sys.path.append("../myTools/rb_tools")
from RbRuntimeTool import Rb_infer, Rb_initial_net, resize_img, pytorch_load_state_dict_no_parallel
from onnxruntime.datasets import get_example
import cv2
import numpy as np
import torch
import models
import onnxruntime



class FeatureExtractor(torch.nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            if name is "fc": x = x.view(x.size(0), -1)
            x = module(x)  # last layer output put into current layer input
            print(name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs







proto = "./resnet50_sg.pbtxt"
coeff = "./resnet50_float_coeff/coeff_little/"
image_path = "./img652.jpg"
onnx_model_path = "/workplace/workspace/filter-pruning-geometric-median/resnet50.onnx"
ckpt="./scripts/snapshots/resnet50-rate-0.7/best.resnet50.GM_0.7_76.82.pth.tar"

img_ori = cv2.imread(image_path)  # BGR
# img = image_preprocess(img_ori)
img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
img, ratio_h, ratio_w = resize_img(img,224,224)
img = img - np.array([123.68, 116.78, 103.94],dtype=np.float32)
img_batch = np.expand_dims(img, 0)


# rb-runtime
net = Rb_initial_net(proto,coeff)
result_rb = Rb_infer(net,["503"],img_batch)["503"]

img_batch = np.transpose(img_batch,[0,3,1,2])
# onnx-runtime
example_model = get_example(onnx_model_path)
sess = onnxruntime.InferenceSession(example_model)
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)
input_type = sess.get_inputs()[0].type
print("Input type  :", input_type)
output_name = sess.get_outputs()[0].name
print("Output name  :", output_name)
output_shape = sess.get_outputs()[0].shape
print("Output shape :", output_shape)
output_type = sess.get_outputs()[0].type
print("Output type  :", output_type)

print("Input data shape{}".format(img_batch.shape))
assert(list(input_shape) == list(img_batch.shape))
result_onnx = sess.run([output_name], {input_name: img_batch})



# pytorch

model = models.__dict__["resnet50"]()
checkpoint = torch.load(ckpt,map_location='cpu')
best_prec1 = checkpoint['best_prec1']
model = pytorch_load_state_dict_no_parallel(model,checkpoint)
model.eval()
img_batch = torch.FloatTensor(img_batch)
with torch.no_grad():
    result_torch = model(img_batch)
    result_torch = result_torch.numpy()


print("max rb-onnx:{}".format(np.max(result_rb-result_onnx)))
print("max onnx-torch:{}".format(np.max(result_onnx-result_torch)))

