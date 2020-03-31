import torch
import numpy as np
import torchvision
from tensorboardX.pytorch_graph import *
from torch._C import Graph,Node,Type
import logging, re

def scope2name(scope):
    patern = r"(?<=\[)\w+(?=\])"
    return '.'.join(re.findall(patern,scope))


def shortcut_recursive_backward_search(node:torch._C.Node,is_head = True):
    """
    the function is for searching structure (convolution specifically) from shortcut-add node forwardedly.
    :param node: the shortcut _C.Node (by filtering from jit.trace.graph.nodes())
    :param from_node: node for recursively call
    :return: list of convolution node (which are skipped in pruning procedural)
    """
    assert ((is_head and "add" in node.kind()) or not is_head)

    def recursive(node:torch._C.Node,expect_next_node_kind:list = None):
        inputs_Value = list(node.inputs())
        result_list = []
        for value in inputs_Value:
            if not list(value.uses()):
                continue
            if expect_next_node_kind:
                for exp in expect_next_node_kind:
                    if exp in value.node().kind():
                        logging.debug("next node: {}".format(scope2name(node.scopeName()),scope2name(value.node().scopeName())))
                        result = shortcut_recursive_backward_search(value.node(), False)
                        if result:
                            result_list.extend(result)
                        break
            else:
                result_list.extend(shortcut_recursive_backward_search(value.node(), False))

        return result_list

    def new_structure_error(node_kind):
        raise RuntimeError("New structure in search path which is not config in shortcut forward search: {} ".format(node_kind))

    # first call by add_ node
    # if from_node is None:
    if "add" in node.kind():
        if is_head: return recursive(node, ["add","batch_norm","max_pool2d","relu"])
        else: return []

    elif "convolution" in node.kind():
            return [node]

    elif "batch_norm" in node.kind():
        result = recursive(node, ["add", "batch_norm", "max_pool2d", "relu", "convolution"])
        result.append(node)
        return result

    elif "relu" in node.kind():
        return recursive(node,["add","batch_norm","convolution"])

    elif "batch_norm" in node.kind():
        return recursive(node,["convolution"])

    elif "max_pool2d" in node.kind():
        return recursive(node,["relu","convolution","add","batch_norm"])

    else: new_structure_error(node.kind())


def shortcut_recursive_forward_search(node:torch._C.Node,is_head = True):
    """
    the function is for searching structure (convolution specifically) from shortcut-add node backward.
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
                            logging.debug("next node: {} => {}".format(scope2name(node.scopeName()),scope2name(next_node.scopeName())))
                            result = shortcut_recursive_forward_search(next_node,False)
                            if result:
                                result_list.extend(result)
            else:
                for next_node in [u.user for u in value.uses()]:
                    result_list.extend(shortcut_recursive_forward_search(next_node,False))

        return result_list

    def new_structure_error(node_kind):
        raise RuntimeError("New structure in search path which is not config in shortcut forward search: {} ".format(node_kind))

    # first call by add_ node
    # if from_node is None:
    if "add" in node.kind():
        if is_head: return recursive(node, ["add","batch_norm","max_pool2d","relu","convolution"])
        else: return []

    elif "batch_norm" in node.kind():
        result = recursive(node,["add","batch_norm","max_pool2d","relu","convolution"])
        result.append(node)
        return result

    elif "convolution" in node.kind():
        # find convolution linked by shortcut add, return it.
        return [node]

    elif "relu" in node.kind():
        return recursive(node,["add","batch_norm","max_pool2d","relu","convolution"])

    elif "max_pool2d" in node.kind():
        return recursive(node,["add","batch_norm","max_pool2d","relu","convolution"])

    else: new_structure_error(node.kind())


def conv_deep_n_backward_search(node:torch._C.Node,deep = 1,is_head = True):
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
    def recursive(node: torch._C.Node,re_deep, expect_next_node_kind: list = None):
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
                        logging.debug("next_node: {} => {}".format(scope2name( node.scopeName()),
                                                                   scope2name( value.node().scopeName())))
                        result = conv_deep_n_backward_search(value.node(),re_deep,False)
                        if result:
                            result_list.extend(result)
                if not recursive_flag:
                    logging.warning("the next structure of node {} is not config, which might be a fault.".format(
                        scope2name(node.scopeName())))

            else:
                result = conv_deep_n_backward_search(value.node(), re_deep, False)
                if result:
                    result_list.extend(result)
        return result_list

    def new_structure_error(node_kind):
        raise RuntimeError(
            "New structure in search path which is not config in shortcut forward search: {} ".format(node_kind))

    # first call by add_ node
    # if from_node is None:
    if "convolution" in node.kind():
        if deep == 0:
            logging.info("stop by conv when deep is {}".format(deep))
            return [node]
        else:
            # find convolution linked by shortcut add, return it.
            result = recursive(node, deep-1,["batch_norm", "relu","max_pool2d","add","convolution"])
            if not is_head:
                result.append(node)
            return result

    elif "relu" in node.kind():
        result = recursive(node,deep, ["batch_norm", "relu","max_pool2d","add","convolution"])
        return result


    elif "batch_norm" in node.kind():
        result = recursive(node,deep, ["batch_norm", "relu","max_pool2d","add","convolution"])
        result.append(node)
        return result

    elif "add" in node.kind():
        logging.info("stop by add when deep is {}".format(deep))
        return []


    elif "max_pool2d" in node.kind():
        result = recursive(node,deep, ["batch_norm", "relu","max_pool2d","add","convolution"])
        return result

    else:
        new_structure_error(node.kind())


def conv_deep_n_forward_search(node:torch._C.Node,deep = 1,is_head = True):
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
    def recursive(node: torch._C.Node,deep, expect_next_node_kind: list = None):
        recursive_flag = False
        outputs_Value = list(node.outputs())
        result_list = []
        for value in outputs_Value:
            if not list(value.uses()):
                continue
            if expect_next_node_kind:
                for exp in expect_next_node_kind:
                    for next_node in [u.user for u in value.uses()]:
                        if exp in next_node.kind():
                            recursive_flag = True
                            logging.debug("next_node: {} => {}".format(scope2name(node.scopeName()),scope2name(next_node.scopeName())))
                            result = conv_deep_n_forward_search(next_node,deep,False)
                            if result:
                                result_list.extend(result)
                if not recursive_flag:
                    logging.warning("the next structure of node {} is not config, which might be a fault.".format(scope2name(node.scopeName())))
            else:
                for next_node in [u.user for u in value.uses()]:
                    result_list.extend(conv_deep_n_forward_search(next_node,deep, False))

        return result_list

    def new_structure_error(node_kind):
        raise RuntimeError(
            "New structure in search path which is not config in shortcut forward search: {} ".format(node_kind))

    # first call by add_ node
    # if from_node is None:
    if "convolution" in node.kind():
        if deep == 0:
            logging.info("stop by conv when deep is {}".format(deep))
            return [node]
        else:
            # find convolution linked by shortcut add, return it.
            result = recursive(node, deep-1,["batch_norm", "relu","max_pool2d","add","convolution"])
            if not is_head:
                result.append(node)
            return result

    elif "relu" in node.kind():
        result = recursive(node,deep, ["batch_norm", "relu","max_pool2d","add","convolution"])
        return result


    elif "batch_norm" in node.kind():
        result = recursive(node,deep, ["batch_norm", "relu","max_pool2d","add","convolution"])
        result.append(node)
        return result

    elif "add" in node.kind():
        logging.info("stop by add when deep is {}".format(deep))
        return []


    elif "max_pool2d" in node.kind():
        result = recursive(node,deep, ["batch_norm", "relu","max_pool2d","add","convolution"])
        return result


    else:
        new_structure_error(node.kind())






