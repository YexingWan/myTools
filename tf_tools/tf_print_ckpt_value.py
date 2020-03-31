import os  
from tensorflow.python import pywrap_tensorflow  
model_dir="./checkpoints/landmark-face-pose/model-9000.ckpt" #checkpoint的文件位置  
# Read data from checkpoint file  
reader = pywrap_tensorflow.NewCheckpointReader(model_dir)  
var_to_shape_map = reader.get_variable_to_shape_map()  
# Print tensor name and values  
for key in var_to_shape_map:  
    print("tensor_name: ", key)  #输出变量名  
    #print(reader.get_tensor(key))   #输出变量值  