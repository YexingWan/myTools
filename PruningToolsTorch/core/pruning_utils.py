import re

def print_list_node(nodes,title = "Nodes"):
    print("========{}========".format(title))
    print("\n".join(["{}\t\t=> {}".format(scope2name(n.scopeName()),n) for n in nodes]))

def scope2name(scope):
    patern = r"(?<=\[)\w+(?=\])"
    return '.'.join(re.findall(patern,scope))


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
