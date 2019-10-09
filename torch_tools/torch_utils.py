from collections import OrderedDict
import torch



# refer: https://sparkydogx.github.io/2018/08/08/pytorch-dataparallel-problem/
# while use DataParallel to train and save state_dict, the name of parameter will have extra ':module'
def pytorch_load_state_dict_no_parallel(model, state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def pytorch_flatten_print_model(model:torch.nn.Module):
    for m_name,m in model.named_modules():
        if not list(m.children()):
            print("[", m_name, "] -> ",m)
            print("[parameters] -> ")
            for idx, (p_name, p) in enumerate(m.named_parameters()):
                print(idx, "-> [", p_name, "] :", p.size())
            print("="*100)
