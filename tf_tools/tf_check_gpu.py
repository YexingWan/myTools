from tensorflow.python.client import device_lib
import os

os.environ['CUDA_VISIBLE_DEVICES']="0,1,3"

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

for d in get_available_gpus():
    print(d)