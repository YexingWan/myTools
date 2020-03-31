# Pruning torch
插入式的torch模型pruning工具,当前支持torchvision中所有backbone网络的剪枝,当前实现的算法有:
* [FPGM](https://arxiv.org/abs/1811.00250)
* [ThiNet](https://arxiv.org/abs/1707.06342)
* [APoZ](https://arxiv.org/pdf/1607.03250.pdf)
* [FilterNorm](https://openreview.net/pdf?id=rJqFGTslg)

Alg TODO List:
 - [ ] Oracel
 - [ ] Approximate Oracel
 - [ ] DCP
 - [ ] RandomPruning
 - [ ] NISP
 - [ ] ACM 

 
## 模型支持

当前用于测试的模型主要是torchvision.model中的模型,具体可以参考项目根目录的test_MaskImpt.py

对于实现较为规范的模型有较好的支持,但由于torch动态图的原因,模型解析较为麻烦,还存在已测和未测的bug.由于torch.jit的支持问题,**如果模型的子模块使用到nn.ModuleList,则会出现模型解析错误的问题,因此当前还未支持含ModuleList的模型**.

ModuleList的支持会在下个版本作支持,具体开发思路可以参考develop.md开发文档

## 使用方法

使用剪枝接口比较简单,详细例子可以参考test_MaskImpt.py,这里结合代码简单做必要说明.

```python

from torchvision import models
from MaskImpt.MaskImpt import MaskFPGM


# 首先,初始化模型,并load权重,这里直接用trochvision的模型作为标准
model = models.resnet50(pretrained=True)

# 一些简单的pruning方法(特别是filter pruning)只需要在eval模式下进行
# 而如果需要用到梯度的算法,则需要用train模式,比如oracle,pruning算法内部会再次确保模式的正确
model.eval() 
use_cuda = True
pre_layer = False
pruned_rate = 0.3

# 第一步,初始化Mask对象,每个已实现的Mask类子类都是一个实现好的pruning算法
# 不同的Mask子类根据算法的需要会有不一样的初始化参数需求,部分需要提供对应的dataloader以供训练
mask = MaskFPGM(model,
                pre_layer=pre_layer,
                rate_dist_per_layer=pruned_rate,
                use_cuda=use_cuda,
                input_shape=tuple([1,3,224,224]) # torch是BCHW
                )

# 第二步,初始化参数,mask,hook等操作          
mask.make()

# 第三步,用剪枝算法进行soft-pruning
# 此时不改变模型总参数量,只会用0/1 mask对权重等置0
# 如果是pre_layer pruning, mask.do_mask()会自动维护pruning顺序
if pre_layer:
    while True:
        is_pruned = mask.do_mask()
        
        # do training ...
        
        if is_pruned:
            print("="*100)
        else: break
else: # 如果是一次性pruning
    mask.do_mask()

# 此时模型已经完成剪枝,但是总参数没变(weight中很多0
# 可以通过mask.if_zero()查看当前模型权重0的比例情况
mask.if_zero()


# 第四步,对模型进行hard_pruning
# 这一步会真实的从权重中切去对应结构(channel,filter,bias,BN的mean/var等)
# hard_pruning后会改变模型结构,真实减少计算量    
pruned_model = mask.generate_pruned_model()
# do training ...



```

## 测试

进行剪枝的过程中和剪枝以后,我们希望做以下几个验证:
1. 正确性: soft-pruning和hard-pruning前后结果是否一致
2. 有效性: hard-pruning后真实减少的MACs和FLOPs以及forward时间

这两个验证任务的实现主要在test_MaskImpt.py中的各个test方法实现.这里给出精简的正确性验证代码,
主要思路是利用hook收集想收集的结果,然后求差.

MACs和FLOPs的计算可以参考并使用[OpCounter](https://github.com/Lyken17/pytorch-OpCounter)

更加详细的验证代码可以参考test_MaskImpt.py中各个test方法代码.
```python
from torchvision import models
from MaskImpt.MaskImpt import MaskFPGM
import torch
import numpy as np


fc_input = []
def fc_hook(module, input, output):
    global fc_input
    fc_input.append(input[0])


model = models.resnet50(pretrained=True)
random_input = np.random.rand(32, 3, 224, 224)
input_tensor = torch.Tensor(random_input).cuda()

mask = MaskFPGM(model,
                pre_layer=False,
                rate_dist_per_layer=0.3,
                use_cuda=True,
                input_shape=tuple([1,3,224,224]) # torch是BCHW
                )
mask.make() # build and compile mask
mask.do_mask() # do soft-pruning

# get the result after soft-pruning (including fc layer if exist)
model = mask.model
model.cuda()
model.eval()

has_handle = False
#将最后一层fc上绑上hook,收集输入tensor
for name, module in model.named_modules():
    if not list(module.children()):
        if isinstance(module, torch.nn.Linear):
            mask_handle = module.register_forward_hook(fc_hook)
            has_handle = True
            break
if has_handle:
    real_pruning_result = fc_input.pop()
    mask_handle.remove()
    

# do hard pruning
model = mask.generate_pruned_model()

# get the result after hard-pruning (including fc layer if exist)
has_handle = False
for name, module in model.named_modules():
    if not list(module.children()):
        if isinstance(module, torch.nn.Linear):
            hard_handle = module.register_forward_hook(fc_hook)
            has_handle = True
            break
if has_handle:
    fake_pruning_result = fc_input.pop()
    mask_handle.remove()

print("error of result of real-pruning and mask-pruning (should be 0):\n {}".format(real_pruning_result-fake_pruning_result))

```


## 算法训练/验证结果

fineturn和验证的结果都是在各自pruning算法的开源项目中完成,并用Mask类复现并验证剪枝正确性.

当前Mask类尚未支持fineturn的管理,后续版本会以callback function的方式进行支持,实验结果也会扩增

| Pruning Alg  | Model  |  pruning-rate  | baseline top1 error  | pruned top1 error | timecost |
| --- | --- | --- | --- | --- | --- |
| FPGM | resnet50v2 | **0.20** | 22.76 | **22.64(-0.13)** | ↓~12% |
| FPGM | resnet50v2 | 0.30 | 22.76 | 23.11(+0.35) | ↓~15% |  
| FPGM| resnet101v1 | 0.3 | 22.63 | 22.67(+0.04) | ↓~17% |
| ThiNet | resnet50v1 | 0.3 | 27.12 | 27.93(+0.92) | ↓~14% |
| APoZ | vgg16 | - | 31.74 | 30.62(+1.12) | - |

 
 
  