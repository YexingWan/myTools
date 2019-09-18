import torch
import models
from collections import OrderedDict
from RbRuntimeTool import pytorch_load_state_dict_no_parallel


state_dict = "/home/yx-wan/newhome/workspace/filter-pruning-geometric-median/scripts/snapshots/resnet50-rate-0.7/best.resnet50.GM_0.7_76.82.pth.tar"
arch = 'resnet50'
model = models.__dict__[arch](pretrained=False).cuda()

def to_ONNX(model, state_dict_path,inputshape):
    checkpoint = torch.load(state_dict_path)
    model = pytorch_load_state_dict_no_parallel(model, checkpoint['state_dict'])
    # model.load_state_dict(checkpoint["state_dict"])
    print("Top 1 precise of model: {}".format(checkpoint['best_prec1']))
    dummy_input = torch.randn(*inputshape).cuda()
    torch.onnx.export(model, dummy_input, "{}.onnx".format(arch), verbose=True)






