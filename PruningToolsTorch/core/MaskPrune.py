import logging
import sys

import numpy as np
import torch
import torch._C
import torch.nn
import re
from copy import deepcopy


#######################

# fp -> filter_pruning
# cp -> channel_pruning

#######################



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
                 model: torch.nn.Module,
                 input_shape: tuple,
                 extra_skip_fp_module_name: list = None,
                 extra_skip_cp_module_name: list = None,
                 skip_tail_conv_fp: int = 1,
                 skip_tail_conv_cp: int = 1,
                 use_cuda: bool = True):
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
        # set check_trace = False for compatibility
        model_trace = torch.jit.trace(model, input_sample, check_trace=False)
        # get the torch._C.Graph of model.
        self.model_graph = model_trace.graph

        self.skip_fp_module = []
        self.skip_cp_module = []
        # {parameter_name: size of parameter}, fill by make()
        self.para_size = {}
        # Pparameter_name: number of weight of parameter}, fill by make()
        self.para_length = {}
        # module_name : [parameter_names], fill by make()
        self.moduels_para_map = {}
        # name of protential masked parameter, fill by make()
        self.mask_name = []
        # name of current masked parameter, fill by make()
        self.cur_mask_name = []
        # mask_name: codebook(*****0/1 1-dem numpy array****), fill by make()
        self.mask = {}
        self.is_make = False
        self.use_cuda = use_cuda


        self.skip_tail_conv_fp = skip_tail_conv_fp,
        self.skip_tail_conv_cp = skip_tail_conv_cp,

        self.extra_skip_cp_module_name = set(
            extra_skip_cp_module_name) if extra_skip_cp_module_name is not None else set()
        self.extra_skip_fp_module_name = set(
            extra_skip_fp_module_name) if extra_skip_fp_module_name is not None else set()

        names = set([name for name, module in model.named_modules() if len(list(module.children())) == 0])
        for m in self.extra_skip_cp_module_name:
            logging.debug("All left module name of module: {}".format(names))
            assert (m in names, "{} in extra_skip_cp_module_name is not in module.")

        for m in self.extra_skip_fp_module_name:
            logging.debug("All left module name of module: {}".format(names))
            assert (m in names, "{} in extra_skip_fp_module_name is not in module.")

        # for general search setting
        self.target_nodes = ["convolution", "aten::batch_norm"]  # the node that will be returned by general_search
        self.deep_sub_nodes = [
            "convolution"]  # the node that deep will -1 in general_search, when deep = 0 search stop. This node will be returned
        self.valid_stop_nodes = ["aten::addmm"]  # the valid stop node. This node will be returned.
        # TODO: support adaptive_avg_pool2d and flatten
        self.ng_nodes = ["aten::add_", "cat", "view", "aten::add", "reshape",
                         "adaptive_avg_pool2d","flatten" ]  # the invalid stop node
        # p.s.: other node will be pass but not return

    # check the exist of ModuleList
    # TODO: support ModelList and ModelDict
    def check_model_validation(self):
        logging.warning("Pruning class has some compatibility problem with model constructed by torch.nn.ModuleList"
                        "Tracing model used torch.jit might raise error or wrong graph, as torch.jit do not parse name of ModuleList and the default name (auto-increase number) of sub-module in list."
                        "Tracing might also be affect and cause wrong result by python logic flood in forward function. [suck feature of dinamic graph :( ]"
                        "So currently we do not support model using ModuleList, till the bug is fixed by torch")

        for name, m in self.model.named_modules():
            if isinstance(m, torch.nn.ModuleList):
                logging.info("ModuleList is currentlt not supported.")
                exit(1)

    def make(self):
        """
        basic initialization procedural
        :return:
        """
        self.make = True
        self.__init_para_length()
        self.__init_moduels_para_map()
        self.__init_mask()
        logging.debug("==========Initialize skip_module_list for filter pruning==========")
        self.skip_fp_module = self.__get_all_fp_skip_module_name()
        logging.debug("==========Initialize skip_module_list foir channel pruning==========")
        self.skip_cp_module = self.__get_all_cp_skip_module_name()

        self.custom_init()
        self.is_make = True
        logging.info("finish make mask")

    # soft-prune the parameter by mask matched by name
    # if you want to implement your own pruning algorithm ,read it !
    @_check_make
    def do_mask(self):
        # custom function
        # update globel(all) mask_name in this pruning iter
        self.update_mask_name()

        # custom function
        # update cur_mask_name in this pruning iter. Only cur_mask_name will be do soft-pruning in this iter
        self.update_cur_mask_name()

        # custom function
        # update value of mask, implement your algorithm in this function
        self.update_mask()



        if len(self.cur_mask_name) == 0 and len(self.mask_name) == 0:
            logging.info("Nothing can be pruned.")
            return False
        else:
            logging.debug("mask_name: {}".format(self.mask_name))

        # named_parameters will return every parameter directly
        for name, item in self.model.named_parameters():
            logging.debug("name_para:{}".format(name))
            if name in self.cur_mask_name:
                logging.debug("Pruning name {}...".format(name))
                a = item.data.view(self.para_length[name])
                b = a * self.__convert2tensor(self.mask[name])
                item.data = b.view(self.para_size[name])
        logging.info("Pruning (by mask) Done")
        print("=" * 100)
        return True
        # refresh mask for current model

    @_check_make
    def if_zero(self):
        """
        print zero status of weight in model
        :return:
        """
        for name, item in self.model.named_parameters():
            if len(self.para_size[name]) == 4:
                a = item.data.view(self.para_length[name])
                b = a.cpu().numpy()
                print("parameter: %s, number of nonzero weight is %d, zero is %d" % (
                    name, np.count_nonzero(b), len(b) - np.count_nonzero(b)))

    @_check_make
    def generate_pruned_model(self):
        """
        The function do hard pruning to model by self.mask
        :return: hard pruned model by mask
        """

        for module_name, module in self.model.named_modules():
            if not list(module.children()):
                logging.debug("pruning sub-module {} ({})...".format(module_name, type(module)))
                # hard pruning for conv2d by mask

                if isinstance(module, torch.nn.Conv2d):
                    left_filter = None
                    for para_name, para in module.named_parameters():
                        # hard pruning the weight
                        if para_name == "weight":
                            para_name = module_name + "." + para_name
                            para.data, left_filter, left_channel = self.__conv_weight_slice_by_mask(para.data,
                                                                                                    self.mask[
                                                                                                        para_name],
                                                                                                    para_name)
                            module.in_channels = para.size()[1]
                            module.out_channels = para.size()[0]
                        # hard pruning the bias if exist
                        elif para_name == "bias":
                            assert (left_filter is not None)
                            para.data = para.data[left_filter]

                # hard pruning for batchnorm2d by mask
                elif isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
                    for para_name, para in module.named_parameters():
                        if para_name == "weight":
                            para_name = module_name + "." + para_name
                            # get left index by mask
                            bn_mask = self.mask[para_name]
                            tep = np.nonzero(bn_mask)
                            assert (len(tep) == 1)
                            keep_index = tep[0]
                            # hard pruning bn
                            self.__bn_postprocess_by_index(module, keep_index)
                            break

                # hard pruning for fc by mask
                elif isinstance(module, torch.nn.modules.linear.Linear):
                    left_filter = None
                    for para_name, para in module.named_parameters():
                        # just do pruning on weight but not bias, as the output channel of fc is not change
                        if para_name == "weight":
                            para_name = module_name + "." + para_name
                            para.data = self.__fc_weight_slice_by_mask(para.data, self.mask[para_name], para_name)
                            module.in_features = para.size()[1]
                            module.out_features = para.size()[0]


                else:
                    logging.debug("module type missmatch.")
        self.__remake()
        return self.model

    def general_search(self, node: torch._C.Node, backward=False, deep=1, is_head=True):
        """
        the recursive search implement
        behaviours is set by self.target_nodes,self.deep_sub_nodes, self.valid_stop_nodes, self.ng_nodes

        :param node: root node, a torch._C.Node object, get by torch._C.Graph.nodes(). the torch._C.Graph get bt torch.jit.trance
        :param backward: boolean, search backward if True, search forward if False
        :param deep: search deep, int, control by self.deep_sub_nodes
        :param is_head: the para used in recursive, dont change
        :return:
        result_list: result node list, defined by self.target_nodes
        result_valid: validation list of search processing, if find ng_node in search path, False will exist in it
        """

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
                        result, valid_end = self.general_search(next_node, backward, deep, False)
                        result_list.extend(result)
                        result_valid.extend(valid_end)
            else:
                for value in inputs_Value:
                    if not list(value.uses()):
                        continue
                    result, valid_end = self.general_search(value.node(), backward, deep, False)
                    result_list.extend(result)
                    result_valid.extend(valid_end)

            return result_list, result_valid

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

    def scope2name(self, scope):
        """
        we can only get scope of a node, change it to name and return.
        As so, ModelList is not support, might be change in future
        :param scope:
        :return:
        """
        patern = r"(?<=\[)\w+(?=\])"
        return '.'.join(re.findall(patern, scope))

    def __convert2tensor(self, x):
        """
        convert input to torch.Tensor
        :param x:
        :return:
        """
        if self.use_cuda:
            x = torch.FloatTensor(x).cuda()
        else:
            x = torch.FloatTensor(x)
        return x

    def __get_all_fp_skip_module_name(self) -> set:
        """
        get all module name that cannot do filter pruning
        :return: set of torch._C.node
        """
        re = set()
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) not in re:
                logging.debug("{} in check.".format(self._scope2name(node.scopeName())))
                # getskip_tail_conv_fp the group number of convolution, if is depth-wise convolution
                if list(node.inputs())[8].toIValue() != 1:
                    logging.debug(
                        "module {} skiped as it is depth-wise convolution".format(self._scope2name(node.scopeName())))
                    result_list, _ = self.general_search(node, backward=True)
                    re.add(self._scope2name(node.scopeName()))
                    for node_skip_dw in result_list:
                        if "convolution" in node_skip_dw.kind():
                            logging.debug("module {} skipped as it is the node backward depth-wise convolution".format(
                                self._scope2name(node_skip_dw.scopeName())))
                            re.add(self._scope2name(node_skip_dw.scopeName()))
                else:
                    result_list, result_valid = self.general_search(node, backward=False)
                    if not all(result_valid):
                        logging.debug("module {} skiped.".format(self._scope2name(node.scopeName())))
                        re.add(self._scope2name(node.scopeName()))
        if self.skip_tail_conv_fp > 0:
            re.update(self.__get_tail_conv(self.skip_tail_conv_fp))

        re.update(self.custom_skip_module_names())
        re.update(self.extra_skip_fp_module_name)
        return re

    def __get_all_cp_skip_module_name(self) -> set:
        """
        get all module name that cannot do channel pruning
        :return: set of torch._C.node
        """
        fp_skip = self.__get_all_fp_skip_module_name()
        re = set()
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) not in re:
                logging.debug("{} in check.".format(self._scope2name(node.scopeName())))
                # get the group number of convolution, if is depth-wise convolution, skip
                if list(node.inputs())[8].toIValue() != 1:
                    logging.debug(
                        "module {} skipped as it is depth-wise convolution".format(self._scope2name(node.scopeName())))
                    re.add(self._scope2name(node.scopeName()))
                    # search the convolution forward and skip (convolution adjacent dw-conv can not prune channel)
                    result_list, _ = self.general_search(node, backward=False)
                    for node_skip_dw in result_list:
                        if "convolution" in node_skip_dw.kind():
                            logging.debug("module {} skipped as it is the node forward depth-wise convolution".format(
                                self._scope2name(node_skip_dw.scopeName())))
                            re.add(self._scope2name(node_skip_dw.scopeName()))
                else:
                    result_list, result_valid = self.general_search(node, backward=True)
                    if not all(result_valid) or (len(result_list) == 0 and len(result_valid) == 0):
                        logging.debug("module {} skiped.".format(self._scope2name(node.scopeName())))
                        re.add(self._scope2name(node.scopeName()))
                        continue
                    # if adjacent convolution can not prune filter,skip node for channel prune
                    for test_node in result_list:
                        if self._scope2name(test_node.scopeName()) in fp_skip:
                            logging.debug("module {} skiped as backward convolution cannot prune filters.".format(
                                self._scope2name(node.scopeName())))
                            re.add(self._scope2name(node.scopeName()))
        if self.skip_tail_conv_cp > 0:
            re.update(self.__get_tail_conv(self.skip_tail_conv_cp))
        re.update(self.custom_skip_module_names())
        re.update(self.extra_skip_cp_module_name)
        return re

    def __get_tail_conv(self, tail: int) -> set:
        """
        get tail convolution node (lease then 'tail' conv node after this conv node) for skip

        :param tail:
        :return:
        """
        tail_conv = set()
        for node in self.model_graph.nodes():
            if "convolution" in node.kind():
                result_list, valid_list = self.general_search(node, deep=tail)
                if all(valid_list):
                    if len([conv_node for conv_node in result_list if "convolution" in conv_node.kind()]) < tail:
                        tail_conv.add(node)
        return tail_conv



    # TODO: support update and maintain menber variable after hard-pruning
    def __remake(self):
        self.__init_para_length()
        self.mask.clear()
        for name in self.mask_name:
            self.mask[name] = np.ones(self.para_length[name])
        self.custom_remake()

    # initialize self.moduels_para_map{ modules_name: [paras_name] }
    def __init_moduels_para_map(self):
        # named_module will return each modules recursively by depth first
        for m_name, m in self.model.named_modules():
            if not list(m.children()):
                p_list = []
                for p_name, p in m.named_parameters():
                    p_list.append(m_name + "." + p_name)
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


    def __conv_weight_slice_by_mask(self, weight: torch.Tensor, mask: np.ndarray, name: str):
        """
        used in hard pruning, do surgery to convolution node
        :param weight:
        :param mask:
        :param name:
        :return:
        """
        size = tuple(self.para_size[name])
        mask = mask.reshape(size)
        left_filter = []
        left_channel = []
        # for filter
        for f in range(size[0]):
            if np.sum(mask[f, :, :, :]) != 0: left_filter.append(f)
        # for filter
        for c in range(size[1]):
            if np.sum(mask[:, c, :, :]) != 0: left_channel.append(c)

        logging.debug("left filter in {} : {}".format(name, left_filter))
        logging.debug("left channel in {} : {}".format(name, left_channel))

        pruned_weight = weight[left_filter, :, :, :]
        pruned_weight = pruned_weight[:, left_channel, :, :]


        logging.debug("pruned {} size: {}".format(name, pruned_weight.size()))
        return pruned_weight, left_filter, left_channel

    def __fc_weight_slice_by_mask(self, weight: torch.Tensor, mask: np.ndarray, name: str):
        """
        used in hard pruning, do surgery to FC(Linear) node

        :param weight: weight Tensor
        :param mask: corresponding mask
        :param name: name of node, for geting size
        :return:
        """
        size = tuple(self.para_size[name])
        mask = mask.reshape(size)
        left_i = []
        logging.debug(size)
        for i in range(size[1]):
            if np.sum(mask[:, i]) != 0: left_i.append(i)
        pruned_weight = weight[:, left_i]
        logging.debug("pruned {} size: {}".format(name, pruned_weight.size()))
        return pruned_weight

    def __bn_postprocess_by_index(self, module: torch.nn.BatchNorm2d, keep_index: list):
        """
        used in hard pruning, do surgery to BN node

        :param module:
        :param pruned_filter_index:
        :return:
        """
        for name, parameter in module.named_parameters():
            parameter.data = parameter.data[keep_index]
        module.num_features = len(keep_index)
        module.running_mean = module.running_mean[keep_index]
        module.running_var = module.running_var[keep_index]




    def custom_remake(self):
        pass

    def custom_init(self):
        """
        custom function
        other parameter initial can be set here, will be run in make()
        :return:
        """
        pass

    def custom_skip_module_names(self) -> set:
        """
        The function can be implemented in subclass. User can implement their own logic to search skip parameters.
        :return: list of parameter which should be skip.
        """
        return set()

    @_check_make
    def update_mask(self):
        """
        custom function
        update value of mask, implement your algorithm in this function
        :return:
        """
        pass

    @_check_make
    def update_mask_name(self):
        """
        custom function
        update globel(all) mask_name in this pruning iter
        :return:
        """

        pass

    @_check_make
    def update_cur_mask_name(self):
        """
        custom function
        update value of mask, implement your algorithm in this function
        :return:
        """
        pass


