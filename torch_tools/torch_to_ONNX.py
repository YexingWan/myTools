import torch
# import models
# from collections import OrderedDict
from torch_utils import pytorch_load_state_dict_no_parallel


# state_dict = "/home/yx-wan/newhome/workspace/filter-pruning-geometric-median/scripts/snapshots/resnet50-rate-0.7/best.resnet50.GM_0.7_76.82.pth.tar"
# arch = 'resnet50'
# model = models.__dict__[arch](pretrained=False).cuda()

def to_ONNX(model,inputshape, state_dict_path=None,output_name="output",cuda=False):
    if state_dict_path is not None:
        checkpoint = torch.load(state_dict_path)
        model = pytorch_load_state_dict_no_parallel(model, checkpoint['state_dict'])
        # model.load_state_dict(checkpoint["state_dict"])
        print("Top 1 precise of model: {}".format(checkpoint['best_prec1']))

    if cuda:
        dummy_input = torch.randn(*inputshape).cuda()
    else:
        dummy_input = torch.randn(*inputshape)
    torch.onnx.export(model, dummy_input, "{}.onnx".format(output_name), verbose=True,opset_version=11)





