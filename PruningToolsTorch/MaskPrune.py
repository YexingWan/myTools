import numpy as np
import torch
import torch.nn
import torch._C
from copy import deepcopy
import logging
import re,sys
sys.path.append(".")
import pruning_utils


def _check_make(func):
    def wrapper(*args, **kwargs):
        if args[0].is_make:
            return func(*args, **kwargs)
        else:
            raise ValueError("You haven't 'make' mask. Use object.make() to initialize the object")

    return wrapper

class Mask:

    def __init__(self,
                 model:torch.nn.Module,
                 input_shape:tuple,
                 extra_skip_fp_module_name: list = None,
                 extra_skip_cp_module_name: list = None,
                 skip_tail_conv_fp: int=1,
                 skip_tail_conv_cp: int =1,
                 use_cuda:bool = True):
        """
        :param model: a torch.nn.Module object for pruning
        :param inputshape: a tuple with 4 length NCHW
        :param use_cuda: whether or not use GPU
        """
        model.eval()
        input_sample = torch.Tensor(np.random.randn(*input_shape))
        self.model = model
        self.check_model_validation()

        if use_cuda:
            input_sample = input_sample.cuda()
        # set check_trace = False for capacity
        model_trace = torch.jit.trace(model, input_sample,check_trace=False)
        # the torch._C.Graph of model.
        self.model_graph = model_trace.graph

        self.skip_fp_module = []
        self.skip_cp_module = []
        # parameter_name: size of parameter
        self.para_size = {}
        # parameter_name: number of weight of parameter
        self.para_length = {}
        # module_name : [parameter_names]
        self.moduels_para_map = {}
        # name of protential masked parameter
        self.mask_name= []
        # name of current masked parameter
        self.cur_mask_name = []
        # mask_name: codebook(0/1 1-dem numpy array)
        self.mask = {}
        self.is_make = False
        self.use_cuda = use_cuda
        self.left_filter = {}
        self.left_channel = {}


        self.skip_tail_conv_fp = skip_tail_conv_fp,
        self.skip_tail_conv_cp = skip_tail_conv_cp,

        self.extra_skip_cp_module_name= set(extra_skip_cp_module_name) if extra_skip_cp_module_name is not None else set()
        self.extra_skip_fp_module_name= set(extra_skip_fp_module_name) if extra_skip_fp_module_name is not None else set()

        names = set([name for name, module in model.named_modules() if len(list(module.children())) == 0])
        for m in self.extra_skip_cp_module_name:
            logging.debug("All left module name of module: {}".format(names))
            assert(m in names, "{} in extra_skip_cp_module_name is not in module.")

        for m in self.extra_skip_fp_module_name:
            logging.debug("All left module name of module: {}".format(names))
            assert(m in names, "{} in extra_skip_fp_module_name is not in module.")







        # general search setting
        self.target_nodes = ["convolution", "aten::batch_norm"] # the node that will be returned by general_search
        self.deep_sub_nodes = ["convolution"] # the node that deep will -1 in general_search, when deep = 0 search stop. This node will be returned
        self.valid_stop_nodes = ["aten::addmm"] # the valid stop node. This node will be returned.
        self.ng_nodes = ["aten::add_","cat","view","aten::add","reshape",] # the invalid stop node
        # p.s.: other node will be pass but not return


    # check the exist of Sequential / ModuleList
    def check_model_validation(self):
        logging.warning("Pruning class has some capacity problem with model constructed by torch.nn.ModuleList"
                        "Tracing model used torch.jit might raise error or wrong graph, as torch.jit do not parse name of ModuleList and the default name (auto-increase number) of sub-module in list."
                        "Tracing might also be affect and cause wrong result by python logic flood in forward function. [suck feature of dinamic graph :( ]"
                        "So currently we do not support model using ModuleList, till the bug is fixed by torch")


        for name, m in self.model.named_modules():
            if isinstance(m,torch.nn.ModuleList):
                logging.info( "ModuleList is not supported: Invalid model for pruning.")
                exit(1)





    def make(self):
        # basic initialization procedural
        self.make = True
        self.__init_para_length()
        self.__init_moduels_para_map()
        self.__init_mask()
        logging.debug("==========Initialize skip_module_list foir fp==========")
        self.skip_fp_module = self.get_all_fp_skip_module_name()
        logging.debug("==========Initialize skip_module_list foir cp==========")
        self.skip_cp_module = self.get_all_cp_skip_module_name()

        self.custom_init()
        self.is_make=True
        logging.info("finish make mask")

    # initialize self.moduels_para_map{ modules_name: [paras_name] }
    def __init_moduels_para_map(self):
        # named_module will return each modules recursively by depth first
        for m_name, m in self.model.named_modules():
            if not list(m.children()):
                p_list = []
                for p_name, p in m.named_parameters():
                    p_list.append(m_name+"."+p_name)
                if p_list:
                    self.moduels_para_map[m_name] = p_list

    # initial self.para_length {paras_name: para_length} and self.para_size {paras_name: torch.Size}
    def __init_para_length(self):
        for name, item in self.model.named_parameters():
            self.para_size[name] = item.size()

        for name in self.para_size.keys():
            for dim in range(0, len(self.para_size[name])):
                if dim == 0:
                    self.para_length[name] = self.para_size[name][0]
                else:
                    self.para_length[name] *= self.para_size[name][dim]

    # initial self.mask_name, self.cur_mask_name [paras_name] and mask {paras_name: np_array with para_length}
    def __init_mask(self):
        self.mask_name = list(self.para_size.keys())
        self.cur_mask_name = deepcopy(self.mask_name)
        for name in self.mask_name:
            self.mask[name] = np.ones(self.para_length[name])

    def custom_init(self):
        pass

    @_check_make
    def update_mask(self):
        pass

    @_check_make
    def update_mask_name(self):
        pass

    @_check_make
    def update_cur_mask_name(self):
        pass

    # mask parameter by mask matched by name
    @_check_make
    def do_mask(self):
        self.update_mask_name()
        self.update_cur_mask_name()
        self.update_mask()
        if len(self.cur_mask_name) == 0 and len(self.mask_name) == 0:
            logging.info("Nothing can be pruned.")
            return
        else:
            logging.debug("mask_name: {}".format(self.mask_name))

        # named_parameters will return every parameter directly
        for name, item in self.model.named_parameters():
            logging.debug("name_para:{}".format(name))
            if name in self.cur_mask_name:
                logging.debug("Pruning name {}...".format(name))
                a = item.data.view(self.para_length[name])
                b = a * self._convert2tensor(self.mask[name])
                item.data = b.view(self.para_size[name])
        logging.info("Pruning (by mask) Done")
        print("="*100)
        # refresh mask for current model

    @_check_make
    def if_zero(self):
        for name, item in self.model.named_parameters():
            if len(self.para_size[name]) == 4:
                a = item.data.view(self.para_length[name])
                b = a.cpu().numpy()
                print("parameter: %s, number of nonzero weight is %d, zero is %d" % (
                    name, np.count_nonzero(b), len(b) - np.count_nonzero(b)))

    def _convert2tensor(self, x):
        if self.use_cuda:
            x = torch.FloatTensor(x).cuda()
        else:
            x = torch.FloatTensor(x)
        return x

    def _scope2name(self,scope):
        return  pruning_utils.scope2name(scope)

    def __conv_weight_slice_by_mask(self, weight:torch.Tensor, mask:np.ndarray, name:str):
        size = tuple(self.para_size[name])
        mask = mask.reshape(size)
        left_filter = []
        left_channel = []
        # for filter
        for f in range(size[0]):
            if np.sum(mask[f,:,:,:]) != 0:left_filter.append(f)
        # for filter
        for c in range(size[1]):
            if np.sum(mask[:,c,:,:]) != 0:left_channel.append(c)

        logging.debug("left filter in {} : {}".format(name,left_filter))
        logging.debug("left channel in {} : {}".format(name,left_channel))


        pruned_weight = weight[left_filter,:,:,:]
        pruned_weight = pruned_weight[:,left_channel,:,:]

        self.left_filter[name] = left_filter
        self.left_channel[name] = left_channel

        logging.debug("pruned {} size: {}".format(name, pruned_weight.size()))
        return pruned_weight,left_filter,left_channel

    def __fc_weight_slice_by_mask(self,weight:torch.Tensor, mask:np.ndarray, name:str):
        size = tuple(self.para_size[name])
        mask = mask.reshape(size)
        left_i = []
        logging.debug(size)
        for i in range(size[1]):
            if np.sum(mask[:,i]) != 0:left_i.append(i)
        pruned_weight = weight[:,left_i]
        logging.debug("pruned {} size: {}".format(name, pruned_weight.size()))
        return pruned_weight

    def __bn_postprocess_by_index(self, module:torch.nn.BatchNorm2d, pruned_filter_index:list):
        for name, parameter in module.named_parameters():
            parameter.data = parameter.data[pruned_filter_index]
        module.num_features=len(pruned_filter_index)
        module.running_mean = module.running_mean[pruned_filter_index]
        module.running_var = module.running_var[pruned_filter_index]

    # get tail conv for skip
    def get_tail_conv(self,tail:int) -> set:
        tail_conv = set()

        for node in self.model_graph.nodes():
            if "convolution" in node.kind():
                result_list, valid_list = self.general_search(node,deep=tail)
                if all(valid_list):
                    if len([conv_node for conv_node in result_list if "convolution" in conv_node.kind()]) < tail:
                        tail_conv.add(node)
        return tail_conv

    def get_all_fp_skip_module_name(self) -> set :
        re = set()
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) not in re:
                logging.debug("{} in check.".format(self._scope2name(node.scopeName())))
                # get the group number of convolution, if is dw
                if list(node.inputs())[8].toIValue() != 1:
                    logging.debug("module {} skiped as it is depth-wise convolution".format(self._scope2name(node.scopeName())))
                    result_list, _ = self.general_search(node,backward=True)
                    re.add(self._scope2name(node.scopeName()))
                    for node_skip_dw in result_list:
                        if "convolution" in node_skip_dw.kind():
                            logging.debug("module {} skipped as it is the node backward depth-wise convolution".format(self._scope2name(node_skip_dw.scopeName())))
                            re.add(self._scope2name(node_skip_dw.scopeName()))
                else:
                    result_list, result_valid = self.general_search(node,backward=False)
                    if not all(result_valid):
                        logging.debug("module {} skiped.".format(self._scope2name(node.scopeName())))
                        re.add(self._scope2name(node.scopeName()))
        if self.skip_tail_conv_fp[0] > 0:
            re.update(self.get_tail_conv(self.skip_tail_conv_fp[0]))
        re.update(self.custom_skip_module_names())
        re.update(self.extra_skip_fp_module_name)
        return re

    def get_all_cp_skip_module_name(self) -> set :
        # get all module name that cannot do filter pruning
        fp_skip = self.get_all_fp_skip_module_name()
        re = set()
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) not in re:
                logging.debug("{} in check.".format(self._scope2name(node.scopeName())))
                # get the group number of convolution, if is dw, skip
                if list(node.inputs())[8].toIValue() != 1:
                    logging.debug("module {} skipped as it is depth-wise convolution".format(self._scope2name(node.scopeName())))
                    re.add(self._scope2name(node.scopeName()))
                    # search the convolution forward and skip (convolution adjacent dw-conv can not prune channel)
                    result_list, _ = self.general_search(node,backward=False)
                    for node_skip_dw in result_list:
                        if "convolution" in node_skip_dw.kind():
                            logging.debug("module {} skipped as it is the node forward depth-wise convolution".format(self._scope2name(node_skip_dw.scopeName())))
                            re.add(self._scope2name(node_skip_dw.scopeName()))
                else:
                    result_list, result_valid = self.general_search(node,backward=True)
                    if not all(result_valid) or (len(result_list) ==0 and len(result_valid) == 0):
                        logging.debug("module {} skiped.".format(self._scope2name(node.scopeName())))
                        re.add(self._scope2name(node.scopeName()))
                        continue
                    # if adjacent convolution can not prune filter, the skip node for channel prune
                    for test_node in result_list:
                        if self._scope2name(test_node.scopeName()) in fp_skip:
                            logging.debug("module {} skiped as backward convolution cannot prune filters.".format(self._scope2name(node.scopeName())))
                            re.add(self._scope2name(node.scopeName()))
        if self.skip_tail_conv_fp[0] > 0:
            re.update(self.get_tail_conv(self.skip_tail_conv_fp[0]))
        re.update(self.custom_skip_module_names())
        re.update(self.extra_skip_cp_module_name)
        return re

    def custom_skip_module_names(self) -> set:
        """
        The function can be implemented in subclass. User can implement their own logic to search skip parameters.
        :return: list of parameter which should be skip.
        """
        return set()

    @_check_make
    def generate_pruned_model(self):
        """
        The function do hard pruning to model by self.mask
        :return: hard pruned model by mask
        """
        self.pruned_model = deepcopy(self.model)

        for module_name, module in self.pruned_model.named_modules():
            if not list(module.children()):
                logging.debug("pruning sub-module {} ({})...".format(module_name,type(module)))
                # hard pruning for conv2d by mask

                if isinstance(module,torch.nn.Conv2d):
                    left_filter = None
                    for para_name,para in module.named_parameters():
                        # hard pruning the weight
                        if para_name == "weight":
                            para_name = module_name+"."+para_name
                            para.data, left_filter, left_channel = self.__conv_weight_slice_by_mask(para.data,self.mask[para_name],para_name)
                            module.in_channels = para.size()[1]
                            module.out_channels = para.size()[0]
                        # hard pruning the bias if exist
                        elif para_name == "bias":
                            assert (left_filter is not None)
                            para.data = para.data[left_filter]

                # hard pruning for batchnorm2d by mask
                elif isinstance(module,torch.nn.modules.batchnorm.BatchNorm2d):
                    for para_name,para in module.named_parameters():
                        if para_name == "weight":
                            para_name = module_name+"."+para_name
                            # get left index by mask
                            bn_mask = self.mask[para_name]
                            tep = np.nonzero(bn_mask)
                            assert (len(tep) == 1)
                            pruned_filter_index = tep[0]
                            # hard pruning bn
                            self.__bn_postprocess_by_index(module,pruned_filter_index)
                            break

                # hard pruning for fc by mask
                elif isinstance(module,torch.nn.modules.linear.Linear):
                    left_filter = None
                    for para_name, para in module.named_parameters():
                        # just do pruning on weight but not bias, as the output channel of fc is not change
                        if para_name == "weight":
                            para_name = module_name+"."+para_name
                            para.data = self.__fc_weight_slice_by_mask(para.data, self.mask[para_name], para_name)
                            module.in_features=para.size()[1]
                            module.out_features=para.size()[0]


                else:
                    logging.debug("module type missmatch.")
        return self.pruned_model

    def general_search(self,node: torch._C.Node, backward = False,deep = 1, is_head=True):
        def recursive(node: torch._C.Node, deep):
            outputs_Value = list(node.outputs())
            inputs_Value = list(node.inputs())
            result_list = []
            result_valid = []
            if not backward:
                for value in outputs_Value:
                    if not list(value.uses()):
                        continue
                    for next_node in [u.user for u in value.uses()]:
                        result, valid_end = self.general_search(next_node,backward, deep, False)
                        result_list.extend(result)
                        result_valid.extend(valid_end)
            else:
                for value in inputs_Value:
                    if not list(value.uses()):
                        continue
                    result, valid_end = self.general_search(value.node(),backward, deep, False)
                    result_list.extend(result)
                    result_valid.extend(valid_end)

            return result_list,result_valid

        for ng_node in self.ng_nodes:
            if ng_node in node.kind():
                logging.debug("stop by {} as ng node when deep is {}".format(node.kind(), deep))
                return [], [False]

        for sub_node in self.deep_sub_nodes:
            if sub_node in node.kind():
                if deep == 0:
                    return [node], [True]
                else:
                    # find convolution linked by shortcut add, return it.
                    result_list, result_valid = recursive(node, deep - 1)
                    if not is_head:
                        result_list.append(node)
                    return result_list, result_valid

        for target_node in self.target_nodes:
            if target_node in node.kind():
                result_list, result_valid = recursive(node, deep)
                if not is_head:
                    result_list.append(node)
                return result_list, result_valid

        for stop_node in self.valid_stop_nodes:
            if stop_node in node.kind():
                return [node], [True]
        result_list, result_valid = recursive(node, deep)
        return result_list, result_valid








    def old_get_all_shortcut_node(self):
        """
        :return: return list of all shortcut "add" torch._C.Node
        """
        shortcut_nodes = []
        for node in self.model_graph.nodes():
            if "add_" in node.kind():
                # print(["{}({},{}):{}".format(n.uniqueName(),n.type().sizes(),node.kind(),n.node()) for n in list(node.outputs())])
                output_value = list(node.outputs())[0]
                if len(output_value.type().sizes()) == 4:
                    shortcut_nodes.append(node)
        # utils.print_list_node(shortcut_nodes, "All shortcut")
        return shortcut_nodes

    def old_get_all_skip_module_names_as_shortcut(self) -> set:
        """
        Many pruning algorithm ignore or skip the convolution directly linked with shortcut of residual block for simple.
        The function find all skipped module through the network structure, and return their name as a list of string.
        :return: list of module_names which should be skip as the shortcut
        """

        shortcut_nodes = self.old_get_all_shortcut_node()
        result_nodes = []

        for node in shortcut_nodes:
            result_nodes.extend(self.old_shortcut_recursive_backward_search(node))
        # utils.print_list_node(result_nodes, "Skip Node")

        skip_module_names = [self._scope2name(node.scopeName()) for node in result_nodes]
        for name in skip_module_names:
            if name not in self.moduels_para_map.keys():
                logging.warning("skip module name {} is not in models.")



        return set(skip_module_names)

    @_check_make
    def old_generate_pruned_model_by_name(self):
        self.pruned_model = deepcopy(self.model)
        para_dict = {}
        for name, parameter in self.pruned_model.named_parameters():
            para_dict[name] = parameter.data

        for name in para_dict.keys():
            if "conv" in name and"weight" in name:
                para_dict[name],left_filter,left_channel = self.__conv_weight_slice_by_mask(para_dict[name],self.mask[name],name)
                bias_name = name.replace("weight","bias")
                if bias_name in para_dict:
                    para_dict[bias_name] = para_dict[bias_name][left_filter]
                bn_weight_name = name.replace("conv","bn")
                if bn_weight_name in para_dict:
                    para_dict[bn_weight_name] = para_dict[bn_weight_name][left_filter]
                    bn_bias_name = bn_weight_name.replace("weight","bias")
                    para_dict[bn_bias_name] = para_dict[bn_bias_name][left_filter]

            elif "downsample.0" in name:
                para_dict[name],left_filter,left_channel = self.__conv_weight_slice_by_mask(para_dict[name],self.mask[name],name)
                bias_name = name.replace("weight","bias")
                if bias_name in para_dict:
                    para_dict[bias_name] = para_dict[bias_name][left_filter]
                bn_weight_name = name.replace("0.weight","1.weight")
                if bn_weight_name in para_dict:
                    para_dict[bn_weight_name] = para_dict[bn_weight_name][left_filter]
                    bn_bias_name = bn_weight_name.replace("weight","bias")
                    para_dict[bn_bias_name] = para_dict[bn_bias_name][left_filter]

            elif "fc" in name and "weight" in name:
                para_dict[name] = self.__fc_weight_slice_by_mask(para_dict[name],self.mask[name],name)

        for name, parameter in self.pruned_model.named_parameters():
            parameter.data = para_dict[name]

        for name, module in self.pruned_model.named_modules():
            if not list(module.children()):
                if isinstance(module,torch.nn.Conv2d):
                    weight = para_dict[name+".weight"]
                    module.in_channels = weight.size()[1]
                    module.out_channels = weight.size()[0]

                elif isinstance(module,torch.nn.BatchNorm2d):
                    weight = para_dict[name+".weight"]
                    module.num_features=weight.size()[0]
                    if "downsample" in name:
                        conv_name = name.replace("downsample.1", "downsample.0") + ".weight"
                    else:
                        conv_name = name.replace("bn","conv")+".weight"
                    module.running_mean = module.running_mean[self.left_filter[conv_name]]
                    module.running_var = module.running_var[self.left_filter[conv_name]]

                elif isinstance(module,torch.nn.Linear):
                    weight = para_dict[name+".weight"]
                    module.in_features=weight.size()[1]
                    module.out_features=weight.size()[0]


        return self.pruned_model

    def old_get_all_skip_module(self) -> set :
        re = self.old_get_all_skip_module_names_as_shortcut()
        re.update(self.custom_skip_module_names())
        return re

    def old_shortcut_recursive_backward_search(self, node: torch._C.Node, is_head=True):
        """
        the function is for searching structure (convolution and batchnorm specifically) from shortcut-add node backwardly.
        :param node: the shortcut _C.Node (by filtering from jit.trace.graph.nodes())
        :param from_node: node for recursively call
        :return: list of convolution node (which are skipped in pruning procedural)
        """
        assert ((is_head and "add" in node.kind()) or not is_head)

        def recursive(node: torch._C.Node, expect_next_node_kind: list = None):
            inputs_Value = list(node.inputs())
            result_list = []
            for value in inputs_Value:
                if not list(value.uses()):
                    continue
                if expect_next_node_kind:
                    for exp in expect_next_node_kind:
                        if exp in value.node().kind():
                            # logging.debug("next node: {} => {}".format(self._scope2name(node.scopeName()),
                            #                                      self._scope2name(value.node().scopeName())))
                            result = self.old_shortcut_recursive_backward_search(value.node(), False)
                            if result:
                                result_list.extend(result)
                            break
                else:
                    result_list.extend(self.old_shortcut_recursive_backward_search(value.node(), False))

            return result_list

        def new_structure_error(node_kind):
            raise RuntimeError(
                "New structure in search path which is not config in shortcut forward search: {} ".format(node_kind))

        # first call by add_ node
        # if from_node is None:
        if "add_" in node.kind():
            if is_head:
                return recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])
            else:
                return []

        elif "batch_norm" in node.kind():
            result = recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])
            result.append(node)
            return result

        elif "convolution" in node.kind():
            # find convolution linked by shortcut add, return it.
            return [node]

        elif "relu" in node.kind():
            return recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])

        elif "max_pool2d" in node.kind():
            return recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])

        else:
            new_structure_error(node.kind())

    def old_shortcut_recursive_forward_search(self, node: torch._C.Node, is_head=True):
        """
        the function is for searching structure (convolution specifically) from shortcut-add node forwardly.
        :param node: the shortcut _C.Node (by filtering from jit.trace.graph.nodes())
        :param from_node: node for recursively call
        :return: list of convolution node (which are skipped in pruning procedural)
        """
        assert ((is_head and "add" in node.kind()) or not is_head)

        def recursive(node: torch._C.Node, expect_next_node_kind: list = None):
            outputs_Value = list(node.outputs())
            result_list = []
            for value in outputs_Value:
                if not list(value.uses()):
                    continue
                if expect_next_node_kind:
                    for exp in expect_next_node_kind:
                        for next_node in [u.user for u in value.uses()]:
                            if exp in next_node.kind():
                                # logging.debug("next node: {} => {}".format(self._scope2name(node.scopeName()),
                                #                                            self._scope2name(next_node.scopeName())))
                                result = self.old_shortcut_recursive_forward_search(next_node, False)
                                if result:
                                    result_list.extend(result)
                else:
                    for next_node in [u.user for u in value.uses()]:
                        result_list.extend(self.old_shortcut_recursive_forward_search(next_node, False))

            return result_list

        def new_structure_error(node_kind):
            raise RuntimeError(
                "New structure in search path which is not config in shortcut forward search: {} ".format(node_kind))

        # first call by add_ node
        # if from_node is None:
        if "add_" in node.kind():
            if is_head:
                return recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])
            else:
                return []

        elif "batch_norm" in node.kind():
            result = recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])
            result.append(node)
            return result

        elif "convolution" in node.kind():
            # find convolution linked by shortcut add, return it.
            return [node]

        elif "relu" in node.kind():
            return recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])

        elif "max_pool2d" in node.kind():
            return recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])

        else:
            new_structure_error(node.kind())

    def old_conv_deep_n_backward_search(self, node: torch._C.Node, deep=1, is_head=True):
        """
        The function is for searching forward n convolution node from current node.
        The whole searching path will be returned.
        The function is recursively called and stopped when deep == 0 and convolution node found.
        Recursive will also stopped when meet "add" and "concat" node(might be updated).
        :param node: the shortcut _C.Node (by filtering from jit.trace.graph.nodes())
        :param from_node: node for recursively call
        :return: list of node on searching path (which are skipped in pruning procedural)
        """
        assert ((is_head and "convolution" in node.kind()) or not is_head)

        # logging.debug("current deep:{}".format(deep))

        def recursive(node: torch._C.Node, re_deep, expect_next_node_kind: list = None):
            inputs_Value = list(node.inputs())
            result_list = []
            recursive_flag = False
            for value in inputs_Value:
                if not list(value.uses()):
                    continue
                if expect_next_node_kind:
                    for exp in expect_next_node_kind:
                        if exp in value.node().kind():
                            recursive_flag = True
                            # logging.debug("next_node: {} => {}".format(self._scope2name(node.scopeName()),
                            #                                            self._scope2name(value.node().scopeName())))
                            result = self.old_conv_deep_n_backward_search(value.node(), re_deep, False)
                            if result:
                                result_list.extend(result)
                            break
                else:
                    recursive_flag = True
                    result = self.old_conv_deep_n_backward_search(value.node(), re_deep, False)
                    if result:
                        result_list.extend(result)

            if not recursive_flag:
                logging.warning(
                    "no adjacent input structure of {} is config, which might be a fault. The kind of adjacent of node is\n {}".format(
                        self._scope2name(node.scopeName()),
                        [value.node().kind() for value in inputs_Value]))

            return result_list

        def new_structure_error(node_kind):
            raise RuntimeError(
                "New structure in search path which is not config in shortcut forward search: {} ".format(node_kind))

        # first call by add_ node
        # if from_node is None:
        if "convolution" in node.kind():
            if deep == 0:
                logging.debug("stop by conv when deep is {}".format(deep))
                return [node]
            else:
                # find convolution linked by shortcut add, return it.
                result = recursive(node, deep - 1,
                                   ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "reshape",
                                    "view"])
                if not is_head:
                    result.append(node)
                return result

        elif "relu" in node.kind():
            result = recursive(node, deep,
                               ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "reshape",
                                "view"])
            return result


        elif "batch_norm" in node.kind():
            result = recursive(node, deep,
                               ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "reshape",
                                "view"])
            result.append(node)
            return result

        elif "add_" in node.kind():
            logging.debug("stop by add when deep is {}".format(deep))
            return []

        elif "max_pool2d" in node.kind():
            result = recursive(node, deep,
                               ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "reshape",
                                "view"])
            return result

        elif "avg_pool2d" in node.kind():
            result = recursive(node, deep,
                               ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "reshape",
                                "view"])
            return result

        elif "reshape" in node.kind():
            result = recursive(node, deep,
                               ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "reshape",
                                "view"])
            return result

        elif "view" in node.kind():
            result = recursive(node, deep,
                               ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "reshape",
                                "view"])
            return result

        else:
            new_structure_error(node.kind())

    def old_conv_deep_n_forward_search(self, node: torch._C.Node, deep=1, is_head=True):
        """
        The function is for searching forward n convolution node from current node.
        The whole searching path will be returned.
        The function is recursively called and stopped when deep == 0 and convolution node found.
        Recursive will also stopped when meet "add" and "concat" node(might be updated).
        :param node: the shortcut _C.Node (by filtering from jit.trace.graph.nodes())
        :param from_node: node for recursively call
        :return: list of node on searching path (which are skipped in pruning procedural)
        """
        assert ((is_head and "convolution" in node.kind()) or not is_head)
        logging.debug("current deep:{}".format(deep))

        def recursive(node: torch._C.Node, deep, expect_next_node_kind: list = None):
            recursive_flag = False
            outputs_Value = list(node.outputs())
            result_list = []
            next_kind = []
            for value in outputs_Value:
                if not list(value.uses()):
                    continue
                if expect_next_node_kind:
                    for exp in expect_next_node_kind:
                        for next_node in [u.user for u in value.uses()]:
                            next_kind.append(next_node.kind())
                            if exp in next_node.kind():
                                recursive_flag = True
                                result = self.old_conv_deep_n_forward_search(next_node, deep, False)
                                if result:
                                    result_list.extend(result)
                else:
                    recursive_flag = True
                    for next_node in [u.user for u in value.uses()]:
                        result = self.old_conv_deep_n_forward_search(next_node, deep, False)
                        result_list.extend(result)

            if not recursive_flag:
                logging.warning(
                    "no adjacent output structure of {} is config, which might be a fault. The kind of adjacent of node is\n {}".format(
                        self._scope2name(node.scopeName()),
                        next_kind))

            return result_list

        def new_structure_error(node_kind):
            raise RuntimeError(
                "New structure in search path which is not config in shortcut forward search: {} ".format(node_kind))

        if "convolution" in node.kind():
            if deep == 0:
                return [node], [True]
            else:
                # find convolution linked by shortcut add, return it.
                result_list = recursive(node, deep - 1,
                                        ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "addmm",
                                         "avg_pool2d", "addmm", "reshape"])
                if not is_head:
                    result_list.append(node)
                return result_list

        elif "relu" in node.kind():
            result_list = recursive(node, deep,
                                    ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "addmm",
                                     "reshape"])
            return result_list


        elif "batch_norm" in node.kind():
            result_list = recursive(node, deep,
                                    ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "addmm",
                                     "reshape"])
            result_list.append(node)
            return result_list

        elif "add_" in node.kind():
            # logging.debug("stop by add when deep is {}".format(deep))
            return []

        elif "max_pool2d" in node.kind():
            logging.debug("input kind of max_pool2d node: {}".format([v.node().kind() for v in node.inputs()]))
            result_list = recursive(node, deep,
                                    ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "addmm",
                                     "reshape", "view"])
            return result_list


        elif "avg_pool2d" in node.kind():
            logging.debug("input kind of avg_pool2d node: {}".format([v.node().kind() for v in node.inputs()]))
            result_list = recursive(node, deep,
                                    ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "addmm",
                                     "reshape", "view"])
            return result_list

        elif "reshape" in node.kind():
            logging.debug("input kind of reshape node: {}".format([v.node().kind() for v in node.inputs()]))
            result_list = recursive(node, deep,
                                    ["batch_norm", "relu", "max_pool2d", "add_", "convolution", "avg_pool2d", "addmm",
                                     "reshape", "view"])
            return result_list

        elif "view" in node.kind():
            logging.debug("input kind of view node: {}".format([v.node().kind() for v in node.inputs()]))
            result_list, result_valid = recursive(node, deep,
                                                  ["batch_norm", "relu", "max_pool2d", "add_", "convolution",
                                                   "avg_pool2d", "addmm", "reshape", "view"])
            return result_list

        elif "addmm" in node.kind():
            logging.debug("stop by fc when deep is {}".format(deep))
            return [node]


        else:
            logging.debug("stop by {} as unknow node when deep is {}".format(node.kind(), deep))
            # return [], [False]
            new_structure_error(node.kind())

    def old_get_related_node_for_pruning_channels(self, conv_module_name):
        """
        :param conv_module_name: the convolution module name that decide to be pruned filters
        :return: A list of module name which should be pruned the channel correspondingly
        """
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) == conv_module_name:
                # the result include related bn (which also should be modified)
                result_list, result_valid = self.general_search(node)
                logging.debug(result_list)
                logging.debug(result_valid)
                assert (all(result_valid))
                return [self._scope2name(node.scopeName()) for node in result_list]

    def old_get_related_node_for_pruning_filters(self, conv_module_name):
        """
        :param conv_module_name: the convolution module name that decide to be pruned channels
        :return: A list of module name which should be pruned the filter correspondingly
        """
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) == conv_module_name:
                # the result include related bn (which also should be modified)
                result_list, _ = self.general_search(node,backward=True)
                return [self._scope2name(node.scopeName()) for node in result_list]


# [2, 3, 5, 8, 9, 11, 12, 14, 15, 17, 18, 21, 23]

# [1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 21, 23]

# [0, 2, 3, 4, 5, 8, 9, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23]

#features.2.conv.2
#features.3.conv.2

#features.3.conv.3


