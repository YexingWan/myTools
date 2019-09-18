import re

def print_list_node(nodes,title = "Nodes"):
    print("========{}========".format(title))
    print("\n".join(["{}\t\t=> {}".format(scope2name(n.scopeName()),n) for n in nodes]))

def scope2name(scope):
    patern = r"(?<=\[)\w+(?=\])"
    return '.'.join(re.findall(patern,scope))