import torch
import torch.nn as nn
import numpy as np
import logging

class cus_model(nn.Module):
    def __init__(self):
        super(cus_model, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3,32,3,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32,31,3,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(31)

        self.conv3 = nn.Conv2d(31,30,3,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(30)

        self.conv4 = nn.Conv2d(30,29,3,padding=1,bias=False)
        self.bn4 = nn.BatchNorm2d(29)

        self.conv5 = nn.Conv2d(30,28,3,padding=1,bias=False)
        self.bn5 = nn.BatchNorm2d(28)

        self.conv6 = nn.Conv2d(30,27,3,padding=1,bias=False)
        self.bn6 = nn.BatchNorm2d(27)

        self.conv7 = nn.Conv2d(27,28,3,padding=1,bias=False)
        self.bn7 = nn.BatchNorm2d(28)

        self.conv8 = nn.Conv2d(56,33,3,padding=1,bias=False)
        self.bn8 = nn.BatchNorm2d(33)

        self.conv9 = nn.Conv2d(33,29,3,padding=1,bias=False)
        self.bn9 = nn.BatchNorm2d(29)



    def forward(self, input):
        re1 = self.conv1(input)
        re1 = self.bn1(re1)
        re1 = self.relu(re1)

        re2 = self.conv2(re1)
        re2 = self.bn2(re2)
        re2 = self.relu(re2)

        re3 = self.conv3(re2)
        re3 = self.bn3(re3)
        re3 = self.relu(re3)

        re4_1 = self.conv4(re3)
        re4_1 = self.bn4(re4_1)
        re4_1 = self.relu(re4_1)

        re4_2 = self.conv5(re3)
        re4_2 = self.bn5(re4_2)
        re4_2 = self.relu(re4_2)

        re4_3 = self.conv6(re3)
        re4_3 = self.bn6(re4_3)
        re4_3 = self.relu(re4_3)

        re4_4 = self.conv7(re4_3)
        re4_4 = self.bn7(re4_4)
        re4_4 = self.relu(re4_4)

        cat = torch.cat((re4_2,re4_4),dim = 1)


        re5 = self.conv8(cat)
        re5 = self.bn8(re5)
        re5 = self.relu(re5)

        re6 = self.conv9(re5)
        re6 = self.bn9(re6)
        re6 = self.relu(re6)

        return re6 + re4_1


res = []
def test_hook(model,input,output):
    global res
    res.append((input[0].shape,output.shape))

model = cus_model()
sample_input = np.random.rand(1,3,64,64)
sample_input = sample_input.astype(np.float32)
tensor = torch.from_numpy(sample_input)

trace = torch.jit.trace(model,(tensor))
graph = trace.graph
print(graph)

for name, module in model.named_modules():
    if isinstance(module,nn.ReLU):
        logging.debug("hook the funtion to module {}".format(name))
        current_hook_handle = module.register_forward_hook(test_hook)
        break

model(tensor)

for p in res:
    print(p)













