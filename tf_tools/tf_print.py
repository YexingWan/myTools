import os
import re

def print_configuration_op(FLAGS):
    print('My Configurations:')
    #pdb.set_trace()
    for name, value in FLAGS.__flags.items():
        value=value.value
        if type(value) == float:
            print(' %s:\t %f'%(name, value))
        elif type(value) == int:
            print(' %s:\t %d'%(name, value))
        elif type(value) == str:
            print(' %s:\t %s'%(name, value))
        elif type(value) == bool:
            print(' %s:\t %s'%(name, value))
        else:
            print(' %s:\t %s' % (name, value))
    #for k, v in sorted(FLAGS.__dict__.items()):
        #print(f'{k}={v}\n')
    print('End of configuration')

def print_checkpoint(checkpoint_path:str):
    os.system("python tf_inspect_checkpoint.py --file_name %s" % checkpoint_path)


def print_variables(vars,filter=None):
    assert (isinstance(vars,dict) or isinstance(vars,list), "expect list or dict, get {}".format(type(vars)))
    if isinstance(vars,dict):
        if filter:
            vars_f = ["{}:{}".format(k, s.shape) for k, s in vars.items() if re.match(filter,k)]
        else:
            vars_f = ["{}:{}".format(k, s.shape) for k, s in vars.items()]

        print("Variable:\n\t{}".format("\n\t".join(vars_f)))
    else:
        if filter:
            vars_f = ["{}:{}".format(v.name[:-2], v.shape) for v in vars if re.match(filter,v.name[:-2])]
        else:
            vars_f = ["{}:{}".format(v.name[:-2], v.shape) for v in vars]
        print("Printing Variable:\n\t{}".format("\n\t".join(vars_f)))



