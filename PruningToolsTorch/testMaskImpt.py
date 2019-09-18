import sys
from MaskImpt import MaskFPGM, MaskThiNet, MaskFilterNorm, old_MaskThiNet, MaskAPoZ
import MaskPrune
import logging
import numpy as np
import torch
from time import time
from torchvision import models, datasets, transforms
sys.path.append("/mnt/newhome/yexing/workspace/PyTorch-YOLOv3")
from models import Darknet




fc_input = []
def fc_hook(module, input, output):
    global fc_input
    fc_input.append(input[0])


def pytorch_flatten_print_model(model):
    for m_name,m in model.named_modules():
        if not list(m.children()):
            print("[", m_name, "] -> ",m)
            print("[parameters] -> ")
            for idx, (p_name, p) in enumerate(m.named_parameters()):
                print(idx, "-> [", p_name, "] :", p.size())
            print("="*100)


def pytorch_flatten_print_parameter(model):
    for p_name,p in model.named_parameters():
        print("[", p_name, "] -> ",p.size())
        print("="*100)


def vis_model(model):
    pytorch_flatten_print_model(model)
    print()
    print("*"*50,"↑ module || ↓ parameters","*"*50)
    print()
    pytorch_flatten_print_parameter(model)


def test_FPGM(model,pre_layer = True,rate = 0.2):
    random_input = np.random.rand(32, 3, 224, 224)
    input_tensor = torch.Tensor(random_input).cuda()
    print("pruning rate: {}".format(rate))

    mask = MaskFPGM(model,pre_layer=pre_layer,rate_dist_per_layer=rate)

    model = mask.model
    model.cuda()
    model.eval()
    s = time()
    for _ in range(10):
        model(input_tensor)
    e = time()
    logging.info("origin model infer time cost: {}s".format((e-s)/10))
    total_ori_para = 0
    for name,p in model.named_parameters():
        total_ori_para = total_ori_para + np.size(p.data.cpu().numpy())



    mask.make()
    if pre_layer:
        for _ in range(len(mask.mask_name)):
            mask.do_mask()
            mask.if_zero()
            print("="*100)
    else:
        mask.do_mask()
    model = mask.model
    model.cuda()
    model.eval()
    for name, module in model.named_modules():
        if not list(module.children()):
            if isinstance(module, torch.nn.Linear):
                mask_handle = module.register_forward_hook(fc_hook)
                break
    fake_pruning_result = model(input_tensor)
    mask_handle.remove()


    model = mask.generate_pruned_model()
    vis_model(model)
    model.cuda()
    model.eval()

    s = time()
    for _ in range(10):
        model(input_tensor)
    e = time()
    for name, module in model.named_modules():
        if not list(module.children()):
            if isinstance(module, torch.nn.Linear):
                hard_handle = module.register_forward_hook(fc_hook)
                break
    real_pruning_result = model(input_tensor)
    hard_handle.remove()


    logging.info("pruned model infer time cost: {}s".format((e-s)/10))


    total_pruned_para = 0
    for name,p in model.named_parameters():
        total_pruned_para = total_pruned_para + np.size(p.data.cpu().numpy())

    logging.info("real pruned rate: {}".format(1-total_pruned_para/total_ori_para))

    logging.info("error of result of real-pruning and mask-pruning (should be 0):\n {}".format(real_pruning_result-fake_pruning_result))

    logging.info("output of fake pruned module:\n {}".format(fake_pruning_result))

    logging.info("output of hard pruned module:\n {}".format(real_pruning_result))

    global fc_input
    if fc_input:
        logging.info("error of input of fc:\n {}".format(fc_input[0]-fc_input[1]))

        logging.info("input of fc in fake pruned module:\n {}".format(fc_input[0]))

        logging.info("input of fc in hard pruned module:\n {}".format(fc_input[1]))


def test_ThiNet(model):
    mask = old_MaskThiNet(model,"/home/yx-wan/newhome/Imagenet/hxfan_val/")
    # mask.do_mask()
    mask.make()

    for _ in range(len(mask.mask_name)):
        mask.do_mask()
        mask.if_zero()
        print("="*100)

    model = mask.generate_pruned_model()
    vis_model(model)


def test_maskprune(mask:MaskPrune.Mask, input_tensor:torch.Tensor, use_cuda = True,pre_layer = False, *kwargs):

    # test origin model
    model = mask.model
    if use_cuda:
        model.cuda()
        input_tensor.cuda()
    model.eval()
    s = time()
    for _ in range(50):
        model(input_tensor)
    e = time()
    origin_cost = (e-s)/50
    total_ori_para = 0
    for name,p in model.named_parameters():
        total_ori_para = total_ori_para + np.size(p.data.cpu().numpy())

    # do soft-pruning
    mask.make()
    if pre_layer:
        if len(mask.mask_name) == 0:
            logging.info("Nothing can be pruned.")
            exit(1)
        for _ in range(len(mask.mask_name)):
            mask.do_mask()
            mask.if_zero()
            print("="*100)
    else:
        mask.do_mask()


    model = mask.model
    if use_cuda:
        model.cuda()
    model.eval()
    has_handle = False
    for name, module in model.named_modules():
        if not list(module.children()):
            if isinstance(module, torch.nn.Linear):
                mask_handle = module.register_forward_hook(fc_hook)
                has_handle = True
                break
    fake_pruning_result = model(input_tensor)
    if has_handle:
        mask_handle.remove()

    # do hard pruning
    model = mask.generate_pruned_model()
    vis_model(model)
    if use_cuda:
        model.cuda()
    model.eval()
    s = time()
    for _ in range(50):
        model(input_tensor)
    e = time()
    has_handle = False
    for name, module in model.named_modules():
        if not list(module.children()):
            if isinstance(module, torch.nn.Linear):
                hard_handle = module.register_forward_hook(fc_hook)
                has_handle = True
                break
    real_pruning_result = model(input_tensor)

    if has_handle:
        hard_handle.remove()


    # logging info
    logging.info("pruned model infer 1 batch with shape {} cost: {}s".format(input_tensor.shape,(e-s)/10))
    logging.info("origin model infer 1 batch with shape {} cost: {}s".format(input_tensor.shape,origin_cost))


    total_pruned_para = 0
    for name,p in model.named_parameters():
        total_pruned_para = total_pruned_para + np.size(p.data.cpu().numpy())

    logging.info("real pruned rate: {}".format(1-total_pruned_para/total_ori_para))

    logging.info("error of result of real-pruning and mask-pruning (should be 0):\n {}".format(real_pruning_result-fake_pruning_result))

    logging.info("output of fake pruned module:\n {}".format(fake_pruning_result))

    logging.info("output of hard pruned module:\n {}".format(real_pruning_result))

    global fc_input
    if fc_input:
        logging.info("error of input of fc:\n {}".format(fc_input[0]-fc_input[1]))

        logging.info("input of fc in fake pruned module:\n {}".format(fc_input[0]))

        logging.info("input of fc in hard pruned module:\n {}".format(fc_input[1]))




logging.basicConfig(level=logging.DEBUG)
# create torch model
model = models.resnet50()                               # success
# model = models.vgg19_bn()                               # success
# model = models.densenet121()                            # success
# model = models.inception_v3()                           # success: relatively small loss (e6 with 1 / e8 with e3)
# model = models.resnext50_32x4d()                        # success
# model  = models.mobilenet_v2()                          # success: cannot be pruned by APoZ (used ReLU6, depth-wise)
# model = models.googlenet()                              # success:
# model = models.shufflenet_v2_x2_0()                     # success: cannot be pruned by APoZ (depth-wise)
# model = models.alexnet()                                # success: add view into invalid stop node (can be optimized)
# model = models.squeezenet1_1()                          # success
# random_input = np.random.rand(1, 3, 224, 224)


################################################################
# create Yolov3
weights_path = "/home/yx-wan/newhome/workspace/PyTorch-YOLOv3/weights/yolov3.weights"
model_def = "/home/yx-wan/newhome/workspace/PyTorch-YOLOv3/config/yolov3.cfg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet(model_def).to(device)
if weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(weights_path))
random_input = np.random.rand(1, 3, 416, 416)

################################################################3

model.eval()

use_cuda = True
pre_layer = False
pruned_rate = 0.3
left_rate = 0.7
input_tensor = torch.Tensor(random_input)
val_dir = "/home/yx-wan/newhome/Imagenet-pic/hxfan_val/"
if use_cuda:
    input_tensor = input_tensor.cuda()
    model = model.cuda()
data_loader = torch.utils.data.DataLoader(datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])), batch_size=2, shuffle=True,num_workers=4, pin_memory=True)



mask = MaskFPGM(model,
                pre_layer=pre_layer,
                rate_dist_per_layer=pruned_rate,
                use_cuda=use_cuda,
                input_shape=tuple(random_input.shape))

# mask = MaskFilterNorm(model, norm=2,
#                       keep_rate=left_rate,
#                       pre_layer=pre_layer,
#                       use_cuda=use_cuda,
#                       input_shape=tuple(random_input.shape))

# mask = MaskThiNet(model,dataloader=data_loader,
#                   input_shape=tuple(random_input.shape),
#                   current_epoch=0,use_cuda=use_cuda,ratio=left_rate)
#
# mask = MaskAPoZ(model,dataloader=data_loader,
#                   input_shape=tuple(random_input.shape),
#                   current_epoch=0,use_cuda=use_cuda,ratio=left_rate)


# test_FPGM(model,pre_layer = False,rate = 0.3)
# test_ThiNet(model)
test_maskprune(mask,input_tensor,pre_layer = pre_layer)

