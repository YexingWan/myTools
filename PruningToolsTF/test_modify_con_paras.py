import tensorflow as tf
import numpy as np

def conv_bn_relu(output_channel, strides = 1):
    seq = tf.keras.Sequential()
    conv = tf.keras.layers.Conv2D(32,3,strides=(2,2),)
    seq.add(conv)
    bn = tf.keras.layers.BatchNormalization()
    seq.add(bn)
    relu = tf.keras.layers.ReLU()
    seq.add(relu)
    return seq


def build_model():
    input = tf.placeholder(dtype=tf.float32,shape=(None,224,224,3),name="input")
    seq = tf.keras.Sequential()
    seq.add(conv_bn_relu(32))
    seq.add(conv_bn_relu(64,2))
    seq.add(conv_bn_relu(32))
    seq.add(conv_bn_relu(32,2))
    seq.add(tf.keras.layers.Flatten())
    fc = tf.keras.layers.Dense(1000)
    softmax = tf.keras.layers.Softmax()
    seq.add(fc)
    seq.add(softmax)
    result = seq(input)
    return input,result



input,softmax = build_model()
g = tf.get_default_graph()


# Operation
ops = g.get_operations()
for op in ops:
    if op.op_def.name == "Conv2D":
        # new_kernel = tf.Variable(collections="Pruning",name=op.name+"/new_kernel",dtype=tf.float32,expected_shape=(18,3,3,18),)
        conv_node_def = op.node_def
        conv_node_def.name=conv_node_def.name+"/prune"
        conv_op_def = op.op_def

        old_conv_op_inputs = op.inputs


        new_kernel = tf.get_variable(name=op.name+"/new_kernel",
                                     shape = (3,3,old_conv_op_inputs[1].shape[2],32),
                                     dtype=tf.float32,
                                     initializer=tf.initializers.random_normal,
                                     trainable=True)
        tf.add_to_collection("Pruning", new_kernel)
        new_kernel_tensor = g.get_tensor_by_name(new_kernel.op.name + '/read:0')

        new_conv_op = tf.Operation(node_def = conv_node_def,g=g,op_def=op.op_def,inputs=[old_conv_op_inputs[0],new_kernel_tensor])
        #
        #
        # new_conv_op._update_input(1,new_kernel_tensor)
        # new_conv_op._update_input(0,op.inputs[0])
        new_conv_op_output = new_conv_op.outputs[0]
        old_conv_op_output = op.outputs[0]

        old_conv_op_output_consumers = old_conv_op_output.consumers()
        consumers_indices = {}
        for c in old_conv_op_output_consumers:
            consumers_indices[c] = [
                i for i, t in enumerate(c.inputs) if t is old_conv_op_output]
        for c in old_conv_op_output_consumers:
            for i in consumers_indices[c]:
                c._update_input(i, new_conv_op_output)
        # for i,t in enumerate(old_op.inputs):
        #     new_op._update_input(i,t)

        # op._update_input(1,new_kernel_tensor)

        print(new_kernel_tensor.consumers())

        # old_op_output_consumers = old_op_output.consumers()
        # consumers_indices = {}
        # for c in old_op_output_consumers:
        #     consumers_indices[c] = [
        #         i for i, t in enumerate(c.inputs) if t is old_op_output]
        # for c in old_op_output_consumers:
        #     for i in consumers_indices[c]:
        #         c._update_input(i, new_op_output)
        # for i,t in enumerate(old_op.inputs):
        #     new_op._update_input(i,t)



        # old_op = g.get_operation_by_name(g.get_operation_by_name(op.node_def.input[1]).node_def.input[0])
        # node_def = old_op.node_def
        print(">>>>>>>>>>>>old_nodedef<<<<<<<<<<<<")
        # print(node_def)
        # print("inputList:{}".format(list(old_op.inputs)))
        # print("oututList:{}".format(list(old_op.outputs)))
        # print(old_op.outputs[0].consumers())


        # node_def.attr["shape"].shape.dim[3].size = 18
        # node_def.name=node_def.name+"/kernel_new"
        # print(str(node_def.attr["_class"].list.s[0],encoding="UTF-8")+"_new")
        # node_def.attr["_class"].list.s[0] = bytes(str(node_def.attr["_class"].list.s[0],encoding="UTF-8")+"_new",encoding="UTF-8")
        # node_def.attr["shared_name"].s = bytes(str(node_def.attr["shared_name"].s,encoding="UTF-8")+"/kernel_new",encoding="UTF-8")

        # node_def.attr["shared_name"].value.s = node_def.attr["shared_name"].list.s+"_new"


        print(">>>>>>>>>>>>new_nodedef<<<<<<<<<<<<")

        # print(node_def)

        print(">>>>>>>>>>>>new_op<<<<<<<<<<<<")

        # new_op = tf.Operation(node_def = node_def,
        #                       g = g,
        #                       op_def=old_op.op_def,
        #                       control_inputs=old_op.control_inputs)
        # new_op_output =  new_op.outputs[0]
        # old_op_output =  old_op.outputs[0]

        # old_op_output_consumers = old_op_output.consumers()
        # consumers_indices = {}
        # for c in old_op_output_consumers:
        #     consumers_indices[c] = [
        #         i for i, t in enumerate(c.inputs) if t is old_op_output]
        # for c in old_op_output_consumers:
        #     for i in consumers_indices[c]:
        #         c._update_input(i, new_op_output)
        # for i,t in enumerate(old_op.inputs):
        #     new_op._update_input(i,t)
        #
        #
        # print(new_op.node_def)
        # print("inputList:{}".format(list(new_op.inputs)))
        # print("oututList:{}".format(list(new_op.outputs)))
        # print(new_op.outputs[0].consumers())


        # setattr(g.get_operation_by_name(g.get_operation_by_name(op.node_def.input[1]).node_def.input[0]).node_def.attr["shape"].ListFields()[0][1].dim[1],'size',5)


        print("="*60)

# print("Operation:\n\t"+"\n\t".join(["\n{}:\n{}\n----------------------------------------------------------\n".format(o.name,g.get_operation_by_name(g.get_operation_by_name(o.node_def.input[1]).node_def.input[0])) for o in ops if o.op_def.name=="Conv2D"]))
# print(">"*30+"op_dict"+"<"*30)

print([a for a in dir(ops[0]) if "__" not in a])


tf.summary.FileWriter(graph=g,logdir="./")

# sess = tf.Session(graph=g)
# tf.global_variables_initializer().run(session=sess)
# dummy_input = np.random.random((1,224,224,3))
# pred = sess.run([softmax],feed_dict={input:dummy_input})
# print(pred[0])
# print(pred[0].shape)



# # Paras
# all_pars = g.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
# print("Parameter:\n\t"+"\n\t".join(["{}:{}".format(v.name,v) for v in all_pars]))
# print(">"*30+"para_dict"+"<"*30)
# print([a for a in dir(all_pars[0]) if "__" not in a])
