import numpy as np
import torch
import torch.utils
from scipy.spatial import distance
from MaskPrune import Mask, _check_make
import logging
from copy import deepcopy
from torchvision import datasets, transforms
from collections import defaultdict


# Done
class old_MaskFPGM(Mask):

    def __init__(self,model,arch="resnet50",
                 rate_keep_norm_per_layer = 1.0, rate_dist_per_layer = 0.3,
                 pre_layer = False,use_cuda = True,input_shape = (1,3,224,224)
                 # , ignore_downsample = True,ignore_all_shortcut = True
                 ):

        Mask.__init__(self,model,input_shape,use_cuda)
        self.arch = arch
        self.rate_keep_norm = rate_keep_norm_per_layer
        self.rate_dist = rate_dist_per_layer
        self.pre_layer = pre_layer
        # self.ignore_downsample = ignore_downsample
        # self.ignore_all_shortcut = ignore_all_shortcut
        self.para_after_map = {}
        self.extra_cur_mask_name = set()


    def custom_init(self):

        self.mask_name = [n for n in self.mask_name if
                          ("conv" in n or "fc" in n) and
                          "weight" in n ]

        self.cur_mask_name = deepcopy(self.mask_name)

        name_before = None
        # initial self.para_after_map for channel pruning
        for name in self.cur_mask_name:
            if name_before is None:
                name_before = name
            else:
                self.para_after_map[name_before] = name
                name_before = name


    @_check_make
    def update_cur_mask_name(self):
        if self.pre_layer:
            self.__pre_layer_prune()
            logging.debug("Current cur_mask_name is {}".format(self.cur_mask_name))
        else:
            # reset current mask name
            self.cur_mask_name = deepcopy(self.mask_name)


    @_check_make
    def update_mask(self):

        for name, item in self.model.named_parameters():
            if name in self.cur_mask_name:


                # # if skip all conv3 and shortcut
                # if self.ignore_all_shortcut and "conv3" in name:
                #     continue

                if "fc" in name or "conv3" in name:
                    continue

                # if ("conv3" in name) and ("0.conv3" not in name) :
                #     continue
                #
                # # if the parameter is the last conv in layer and not skip_donwsample
                # if not self.ignore_downsample and "0.conv3" in name:
                #     logging.debug("take downsample into account.")
                #     down_name = name.replace("conv3", "downsample.0")   # construct corresponded downsample paras name
                #     for name_down, item_down in self.model.named_parameters():
                #         if name_down == down_name:
                #             logging.debug("dealing with paras {} and {}".format(name,name_down))
                #             cat_tensor = torch.cat([item.data,item_down.data],dim = 1) # concatenation current para and shortcut
                #             logging.debug("The concat tensor size is {}".format(cat_tensor.size()))
                #
                #             # built temporary para_length and mask for using __get_filter function
                #             self.para_length["tem_para"] = self.para_length[down_name]+self.para_length[name]
                #             self.mask["tem_para"] = np.ones(self.para_length["tem_para"])
                #             _,filter_index = self.__get_filter(cat_tensor,"tem_para")
                #             self.para_length.pop("tem_para")
                #             self.mask.pop("tem_para")
                #
                #             # build codebook by filter_index list
                #             codebook_down = self.mask[down_name]
                #             codebook_conv = self.mask[name]
                #             kernel_length_down = self.para_size[down_name][1]*self.para_size[down_name][2]*self.para_size[down_name][3]
                #             kernel_length_conv = self.para_size[name][1]*self.para_size[name][2]*self.para_size[name][3]
                #             for x in range(0, len(filter_index)):
                #                 codebook_down[filter_index[x] * kernel_length_down: (filter_index[x] + 1) * kernel_length_down] = 0
                #                 codebook_conv[filter_index[x] * kernel_length_conv: (filter_index[x] + 1) * kernel_length_conv] = 0
                #
                #             self.mask[down_name] = codebook_down
                #             self.mask[name] = codebook_conv
                #             self.extra_cur_mask_name.add(down_name)
                #             # if parameter related to next parameter channel pruning
                #             self.__after_channel_pruning(name,filter_index)
                #     continue

                self.mask[name], filter_index = self.__get_filter(item.data,name)
                # if parameter related to next parameter channel pruning
                self.__after_channel_pruning(name, filter_index)

                # if the parameter is the last conv in layer and skip_donwsample
                # if self.ignore_downsample and "0.conv3.weight" in name:
                #     logging.debug("skip downsample and prune it be same as linked conv")
                #     down_name = name.replace("conv3", "downsample.0")
                #     codebook = self.mask[down_name]
                #     logging.debug("dealing with paras {}".format(down_name))
                #     kernel_length = self.para_size[down_name][1]*self.para_size[down_name][2]*self.para_size[down_name][3]
                #     for x in range(0, len(filter_index)):
                #         codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0
                #     self.mask[down_name] = codebook
                #     self.extra_cur_mask_name.add(down_name)

        self.cur_mask_name.extend(list(self.extra_cur_mask_name))
        self.extra_cur_mask_name.clear()


    # update extra_cur_mask_name for extra channel pruning
    def __after_channel_pruning(self,name,filter_index):
        if name in list(self.para_after_map.keys()):

            # set channel pruning for parameter after name
            logging.debug("pair: {}:{}".format(name, self.para_after_map[name]))
            matrix_mask_before = np.reshape(self.mask[self.para_after_map[name]],
                                            tuple(self.para_size[self.para_after_map[name]]))


            if "fc" in self.para_after_map[name]:
                matrix_mask_before[:, filter_index] = 0
            else:
                matrix_mask_before[:, filter_index, :, :] = 0

            self.mask[self.para_after_map[name]] = np.reshape(matrix_mask_before, -1)
            self.extra_cur_mask_name.add(self.para_after_map[name])



            # if parameter related to shortcut channel pruning
            if "conv1" in self.para_after_map[name]:
                logging.debug("dealing with shortcut: {}".format(name))
                # set channel pruning for shortcut
                down_name = self.para_after_map[name].replace("conv1", "downsample.0")
                matrix_mask_after = np.reshape(self.mask[down_name],
                                               tuple(self.para_size[down_name]))
                matrix_mask_after[:, filter_index, :, :] = 0
                self.mask[down_name] = np.reshape(matrix_mask_after, -1)
                self.extra_cur_mask_name.add(down_name)


    # update curr_mask_name for pre-layer pruning
    def __pre_layer_prune(self):
        if self.cur_mask_name:
            self.cur_mask_name.clear()
        self.cur_mask_name.append(self.mask_name.pop(0))
        self.mask_name.append(self.cur_mask_name[0])


    def __get_filter(self,weight_torch,name):
        filter_norm, filter_index_norm = self.__get_filter_norm2_(weight_torch,name)
        logging.debug("finish built norm filter")
        logging.debug("Zero in norm_filter: {}".format(len(filter_norm)-np.count_nonzero(filter_norm)))

        filter_sim, filter_index_norm_sim= self.__get_filter_similar_base_norm_(weight_torch,name)
        logging.debug("finish built GM filter")
        logging.debug("Zero in GM_filter: {}".format(len(filter_sim)-np.count_nonzero(filter_sim)))

        filter = np.logical_and(filter_norm,filter_sim).astype(np.int)
        logging.debug("Zero in filter: {}".format(len(filter) - np.count_nonzero(filter)))
        logging.debug("finish built filter")
        return filter, np.concatenate([filter_index_norm,filter_index_norm_sim])


    def __get_filter_norm2_(self, weight_torch,name):
        assert(len(weight_torch.size()) == 4)
        codebook = self.mask[name]
        filter_pruned_num = int(weight_torch.size()[0] * (1 - self.rate_keep_norm))

        weight_vec = weight_torch.view(weight_torch.size()[0], -1)
        norm2 = torch.norm(weight_vec, 2, 1)
        norm2_np = norm2.detach().cpu().numpy()
        filter_index = norm2_np.argsort()[:filter_pruned_num]
        logging.debug("number of filter pruned as norm: {}".format(filter_pruned_num))
        logging.debug("norm filter index: {}".format(filter_index))
        # norm1 = torch.norm(weight_vec, 1, 1)
        # norm1_np = norm1.cpu().numpy()
        # filter_index = norm1_np.argsort()[:filter_pruned_num]

        kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        for x in range(0, len(filter_index)):
            codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

        return codebook,filter_index


    def __get_filter_similar_base_norm_(self, weight_torch,name):
        assert(len(weight_torch.size()) == 4)
        codebook = self.mask[name]

        norm2_pruned_num= int(weight_torch.size()[0] * (1 - self.rate_keep_norm))
        filter_pruned_num = int(weight_torch.size()[0] * self.rate_dist)

        # fisrt, get the result after pruning by norm2
        weight_vec = weight_torch.view(weight_torch.size()[0], -1)
        norm2 = torch.norm(weight_vec, 2, 1)
        norm2_np = norm2.detach().cpu().numpy()

        norm2_large_index = norm2_np.argsort()[norm2_pruned_num:]
        logging.debug("number of filter left after norm prune: {}".format(len(norm2_large_index)))
        weight_vec_after_norm = torch.index_select(weight_vec, 0, torch.LongTensor(norm2_large_index).cuda()).cpu().numpy()

        # for euclidean distance
        similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')

        # for cos similarity
        # similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')

        similar_sum = np.sum(np.abs(similar_matrix), axis=0)

        # for distance similar: get the filter index with largest similarity / smallest distance
        # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
        distance_small_index = similar_sum.argsort()[:filter_pruned_num]
        distance_small_index = [norm2_large_index[i] for i in distance_small_index]
        logging.debug("GM filter index:{}".format(distance_small_index))

        kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        for x in range(0, len(distance_small_index)):
            codebook[
            distance_small_index[x] * kernel_length: (distance_small_index[x] + 1) * kernel_length] = 0
        return codebook,distance_small_index


# Done
class MaskFPGM(Mask):

    def __init__(self,model,
                 rate_keep_norm_per_layer = 1.0,
                 rate_dist_per_layer = 0.3,
                 pre_layer = False,
                 use_cuda = True,
                 input_shape = tuple([1,3,224,224]),
                 skip_tail_conv_fp: int = 1,
                 skip_tail_conv_cp: int = 1,
                 extra_skip_fp_module_name: list = None,
                 extra_skip_cp_module_name: list = None):

        Mask.__init__(self,
                      model = model,
                      input_shape=input_shape,
                      skip_tail_conv_fp=skip_tail_conv_fp,
                      skip_tail_conv_cp=skip_tail_conv_cp,
                      extra_skip_cp_module_name=extra_skip_cp_module_name,
                      extra_skip_fp_module_name=extra_skip_fp_module_name,
                      use_cuda=use_cuda)
        self.rate_keep_norm = rate_keep_norm_per_layer
        self.rate_dist = rate_dist_per_layer
        self.pre_layer = pre_layer



    def custom_init(self):
        self.mask_name = []
        for node in self.model_graph.nodes():
            if "convolution" in node.kind():
                module_name = self._scope2name(node.scopeName())
                if module_name in self.skip_fp_module:
                    logging.debug("skip module {}".format(module_name))
                    continue
                logging.debug("put parameter name of module {} in self.mask_name".format(module_name+'.weight'))
                self.mask_name.append(module_name+'.weight')
        # logging.debug("Current mask_name is {}".format(self.mask_name))
        self.cur_mask_name = deepcopy(self.mask_name)


    @_check_make
    def update_cur_mask_name(self):
        if self.pre_layer:
            self.__pre_layer_prune()
            logging.debug("Current cur_mask_name is {}".format(self.cur_mask_name))
        else:
            # reset current mask name
            self.cur_mask_name = deepcopy(self.mask_name)


    # update curr_mask_name for pre-layer pruning
    def __pre_layer_prune(self):
        if self.cur_mask_name:
            self.cur_mask_name.clear()
        self.cur_mask_name.append(self.mask_name.pop(0))
        self.mask_name.append(self.cur_mask_name[0])


    @_check_make
    def update_mask(self):
        extra_name = set()
        for module_name, module in self.model.named_modules():
            if not list(module.children()):
                for para_name, para in module.named_parameters():
                    para_name = module_name+"."+para_name
                    if para_name in self.cur_mask_name:
                        new_mask, filter_index = self.__get_filter(para.data, para_name)
                        # update mask of weight, bias, bn and next convolution
                        extra_name.update(self.__update_mask(para_name,module_name,new_mask,filter_index))

        self.cur_mask_name.extend(list(extra_name))


    # update mask of weight, bias, bn and next convolution/fc
    def __update_mask(self,para_name:str,module_name:str,new_mask:np.ndarray,filter_index:np.ndarray) -> set:
        assert(".weight" in para_name)
        new_name = set()

        # weight
        self.mask[para_name] = new_mask
        bias_name = para_name.replace(".weight",".bias")
        # bias
        if bias_name in self.moduels_para_map[module_name]:
            new_name.add(bias_name)
            self.mask[bias_name][filter_index] = 0

        # bn, next_conv/fc
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) == module_name:
                related_nodes,_ = self.general_search(node)
                logging.debug("pruning filter of {} => pruning channel of {}".format(module_name,[self._scope2name(n.scopeName()) for n in related_nodes]))
                assert (len(related_nodes) > 0)
                for r_node in related_nodes:
                    # bn
                    if "batch_norm" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            new_name.add(sub_para_name)
                            self.mask[sub_para_name][filter_index] = 0
                    # next conv
                    elif "convolution" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            if ".weight" in sub_para_name:
                                new_name.add(sub_para_name)
                                matrix_mask_before = np.reshape(self.mask[sub_para_name],
                                                                tuple(self.para_size[sub_para_name]))
                                matrix_mask_before[:, filter_index, :, :] = 0
                                self.mask[sub_para_name] = np.reshape(matrix_mask_before, -1)

                    # TODO:reshape / view fc
                    elif "addmm" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            if ".weight" in sub_para_name:
                                new_name.add(sub_para_name)
                                matrix_mask_before = np.reshape(self.mask[sub_para_name],
                                                                tuple(self.para_size[sub_para_name]))

                                assert (self.para_size[sub_para_name][1] % self.para_size[para_name][0] == 0)
                                num_features_pre_channel = self.para_size[sub_para_name][1] // self.para_size[para_name][0]
                                logging.debug("In fc, {} input feature map 1 channel of last convolution layer.".format(num_features_pre_channel))
                                # get corresponding feature index
                                real_filter_index = np.array([[idx * num_features_pre_channel+j for j in range(num_features_pre_channel)] for idx in filter_index]).flatten()
                                matrix_mask_before[:, real_filter_index] = 0
                                self.mask[sub_para_name] = np.reshape(matrix_mask_before, -1)

        return new_name


    def __get_filter(self,weight_torch,name):
        filter_norm, filter_index_norm = self.__get_filter_norm2_(weight_torch,name)
        logging.debug("finish built norm filter")
        logging.debug("Zero in norm_filter: {}".format(len(filter_norm)-np.count_nonzero(filter_norm)))

        filter_sim, filter_index_norm_sim= self.__get_filter_similar_base_norm_(weight_torch,name)
        logging.debug("finish built GM filter")
        logging.debug("Zero in GM_filter: {}".format(len(filter_sim)-np.count_nonzero(filter_sim)))

        filter = np.logical_and(filter_norm,filter_sim).astype(np.int)
        logging.debug("Zero in filter: {}".format(len(filter) - np.count_nonzero(filter)))
        logging.debug("finish built filter")
        return filter, np.concatenate([filter_index_norm,filter_index_norm_sim])


    def __get_filter_norm2_(self, weight_torch,name):
        assert(len(weight_torch.size()) == 4)
        codebook = self.mask[name]
        filter_pruned_num = int(weight_torch.size()[0] * (1 - self.rate_keep_norm))

        weight_vec = weight_torch.view(weight_torch.size()[0], -1)

        norm2 = torch.norm(weight_vec, 2, 1)
        norm2_np = norm2.detach().cpu().numpy()
        filter_index = norm2_np.argsort()[:filter_pruned_num]
        logging.debug("number of filter pruned as norm: {}".format(filter_pruned_num))
        logging.debug("norm filter index: {}".format(filter_index))
        # norm1 = torch.norm(weight_vec, 1, 1)
        # norm1_np = norm1.cpu().numpy()
        # filter_index = norm1_np.argsort()[:filter_pruned_num]

        kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        for x in range(0, len(filter_index)):
            codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

        return codebook,filter_index


    def __get_filter_similar_base_norm_(self, weight_torch,name):
        assert(len(weight_torch.size()) == 4)
        codebook = self.mask[name]

        norm2_pruned_num= int(weight_torch.size()[0] * (1 - self.rate_keep_norm))
        filter_pruned_num = int(weight_torch.size()[0] * self.rate_dist)

        # fisrt, get the result after pruning by norm2
        weight_vec = weight_torch.view(weight_torch.size()[0], -1)
        norm2 = torch.norm(weight_vec, 2, 1)
        logging.debug("norm shape:{}".format(norm2.shape))
        norm2_np = norm2.detach().cpu().numpy()

        norm2_large_index = norm2_np.argsort()[norm2_pruned_num:]
        logging.debug("number of filter left after norm prune: {}".format(len(norm2_large_index)))
        weight_vec_after_norm = torch.index_select(weight_vec, 0, torch.LongTensor(norm2_large_index).cuda()).cpu().numpy()

        # for euclidean distance
        similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')

        # for cos similarity
        # similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')

        similar_sum = np.sum(np.abs(similar_matrix), axis=0)

        # for distance similar: get the filter index with largest similarity / smallest distance
        # similar_large_index = similar_sum.argsort()[similar_pruned_num:]
        distance_small_index = similar_sum.argsort()[:filter_pruned_num]
        distance_small_index = [norm2_large_index[i] for i in distance_small_index]
        logging.debug("GM filter index:{}".format(distance_small_index))

        kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        for x in range(0, len(distance_small_index)):
            codebook[
            distance_small_index[x] * kernel_length: (distance_small_index[x] + 1) * kernel_length] = 0
        return codebook,distance_small_index


# Done
class old_MaskThiNet(Mask):

    def __init__(self, model, val_dir,input_shape:tuple, current_epoch=0, use_cuda=True, ratio=0.7):
        Mask.__init__(self, model, input_shape, use_cuda)
        self.epoch = -1
        self.current_epoch = current_epoch
        self.current_hook_handle = None
        self.val_dir = val_dir
        self.data_loader = None
        self.ratio = ratio
        self.para_before_map = {}
        logging.debug("ratio: {}".format(self.ratio))

    def custom_init(self):
        self.model.eval()
        self.data_loader = torch.utils.data.DataLoader(datasets.ImageFolder(self.val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])), batch_size=2, shuffle=True,
                                                       num_workers=4, pin_memory=True)
        logging.debug("not filtered mask_name : {}".format(self.mask_name))
        self.mask_name = [n for n in self.mask_name if "conv" in n and "weight" in n and "conv1" not in n]
        logging.debug("filtered mask_name : {}".format(self.mask_name))
        self.epoch = len(self.mask_name) * 2 + 12

        before_name = None

        # initialize before_name_map for channle pruning
        for name, para in self.model.named_parameters():
            if len(self.para_size[name]) == 4:
                if before_name is not None:
                    self.para_before_map[name] = before_name
                    if "downsample" in name or "conv3" in name:
                        before_name = None
                    else:
                        before_name = name
                elif not ("downsample" in name or "conv3" in name):
                    before_name = name
        logging.debug("before map: {}".format(self.para_before_map))

    @_check_make
    def update_cur_mask_name(self):
        global ThiNet_distance
        if self.cur_mask_name:
            self.cur_mask_name.clear()
        if self.current_hook_handle is not None:
            self.current_hook_handle.remove()
        if ThiNet_distance:
            ThiNet_distance.clear()
        # get current para
        self.cur_mask_name.append(self.mask_name.pop(0))
        self.mask_name.append(self.cur_mask_name[0])

        logging.debug("current name: {}".format(self.cur_mask_name))
        logging.debug("in hook")

        for name, module in self.model.named_modules():
            if not list(module.children()):
                if name + ".weight" in self.cur_mask_name:
                    logging.debug("hook the funtion to module {}".format(name))
                    self.current_hook_handle = module.register_forward_hook(ThiNet_hook)

    def update_mask(self):
        global ThiNet_distance
        global tep_output
        for name, para in self.model.named_parameters():
            if name in self.cur_mask_name:
                left_channels = list(range(0, self.para_size[name][1]))
                pruned_channels = []
                # buff origin parameter
                ori_para = deepcopy(para.data.cpu())
                logging.debug("pruning paras name {}".format(name))

                for data, target in self.data_loader:
                    if len(pruned_channels) >= int(self.para_size[name][1] * (1 - self.ratio)):
                        break
                    # get origin output by hook
                    if self.use_cuda:
                        data = data.cuda()
                    self.model(data)
                    # prune all pruned channel
                    for c in pruned_channels:
                        para.data[:, c, :, :] = 0
                    # buff pruned_channels parameter
                    pruned_para = deepcopy(para.data.cpu())
                    # prune left channels one by one and record output by hook
                    for c in left_channels:
                        para.data[:, c, :, :] = 0
                        self.model(data)
                        # reset parameter
                        para.data = deepcopy(pruned_para).cuda()
                    torch.cuda.empty_cache()
                    logging.debug("distance number: {}".format(len(ThiNet_distance)))
                    logging.debug("left_channels number: {}".format(len(left_channels)))
                    assert (len(ThiNet_distance) == len(left_channels))
                    logging.debug("distance {}".format(ThiNet_distance))
                    # get minimum reconstruction loss
                    min_index = np.argmin(ThiNet_distance)
                    logging.debug("less error index of current step is {}".format(min_index))
                    pruned_channels.append(left_channels[min_index.item()])
                    logging.debug("pruned_channel {}".format(pruned_channels))
                    left_channels.pop(min_index.item())
                    logging.debug("left_channel {}".format(left_channels))
                    ThiNet_distance.clear()
                    tep_output = None
                    torch.cuda.empty_cache()
                    # restore parameter
                    para.data = deepcopy(ori_para).cuda()
                logging.debug("pruned clannels are {}".format(pruned_channels))
                # built mask by pruned_channels and pruned_filter for parameter before
                matrix_mask = np.reshape(self.mask[name], tuple(self.para_size[name]))
                before_matrix_mask = np.reshape(self.mask[self.para_before_map[name]],
                                                tuple(self.para_size[self.para_before_map[name]]))
                for c in pruned_channels:
                    matrix_mask[:, c, :, :] = 0
                    before_matrix_mask[c, :, :, :] = 0

                self.mask[name] = np.reshape(matrix_mask, -1)
                self.mask[self.para_before_map[name]] = np.reshape(before_matrix_mask, -1)
                # it will make no fault because self.para_before_map[name] is always occure before name in named_parameters()
                self.cur_mask_name.append(self.para_before_map[name])

    def __objective(self, origin_output, pruned_output):
        return torch.norm(origin_output - pruned_output).item()


# Done
class MaskFilterNorm(Mask):

    def __init__(self,model,
                 norm = 2,
                 keep_rate = 0.8,
                 pre_layer = False,
                 use_cuda = True,
                 input_shape = (1,3,224,224),
                 skip_tail_conv_fp: int = 1,
                 skip_tail_conv_cp: int = 1,
                 extra_skip_fp_module_name: list = None,
                 extra_skip_cp_module_name: list = None
                 ):

        Mask.__init__(self,model = model,
                      input_shape=input_shape,
                      skip_tail_conv_fp=skip_tail_conv_fp,
                      skip_tail_conv_cp=skip_tail_conv_cp,
                      extra_skip_cp_module_name=extra_skip_cp_module_name,
                      extra_skip_fp_module_name=extra_skip_fp_module_name,use_cuda=use_cuda)
        self.rate_keep_norm = keep_rate
        self.pre_layer = pre_layer
        self.norm = norm
        assert (norm == 1 or norm == 2, "The {} class only support norm 1 and norm 2".format(self.__class__))



    def custom_init(self):
        self.mask_name = []
        for node in self.model_graph.nodes():
            if "convolution" in node.kind():
                module_name = self._scope2name(node.scopeName())
                if module_name in self.skip_fp_module:
                    logging.debug("skip module {}".format(module_name))
                    continue
                logging.debug("put parameter name of module {} in self.mask_name".format(module_name+'.weight'))
                self.mask_name.append(module_name+'.weight')
        logging.debug("Current mask_name is {}".format(self.mask_name))
        self.cur_mask_name = deepcopy(self.mask_name)

    @_check_make
    def update_cur_mask_name(self):
        if self.pre_layer:
            self.__pre_layer_prune()
            logging.debug("Current cur_mask_name is {}".format(self.cur_mask_name))
        else:
            # reset current mask name
            self.cur_mask_name = deepcopy(self.mask_name)

    # update curr_mask_name for pre-layer pruning
    def __pre_layer_prune(self):
        if self.cur_mask_name:
            self.cur_mask_name.clear()
        self.cur_mask_name.append(self.mask_name.pop(0))
        self.mask_name.append(self.cur_mask_name[0])

    @_check_make
    def update_mask(self):
        extra_name = set()
        for (module_name, module) in [(n,m) for n,m in self.model.named_modules() if not list(m.children())]:
            for para_name, para in module.named_parameters():
                para_name = module_name+"."+para_name
                if para_name in self.cur_mask_name:
                    new_mask, filter_index = self.__get_filter(para.data, para_name)
                    # update mask of weight, bias, bn and next convolution
                    extra_name.update(self.__update_mask(para_name,module_name,new_mask,filter_index))

        self.cur_mask_name.extend(list(extra_name))

    # update mask of weight, bias, bn and next convolution/fc
    def __update_mask(self,para_name:str,module_name:str,new_mask:np.ndarray,filter_index:np.ndarray) -> set:
        assert(".weight" in para_name)
        new_name = set()

        # weight
        self.mask[para_name] = new_mask
        bias_name = para_name.replace(".weight",".bias")
        # bias
        if bias_name in self.moduels_para_map[module_name]:
            new_name.add(bias_name)
            self.mask[bias_name][filter_index] = 0

        # bn, next_conv/fc
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) == module_name:
                related_nodes,_ = self.general_search(node)
                logging.debug("pruning filter of {} => pruning channel of {}".format(module_name,[self._scope2name(n.scopeName()) for n in related_nodes]))
                assert (len(related_nodes) > 0)
                for r_node in related_nodes:
                    # bn
                    if "batch_norm" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            new_name.add(sub_para_name)
                            self.mask[sub_para_name][filter_index] = 0
                    # next conv
                    elif "convolution" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            if ".weight" in sub_para_name:
                                new_name.add(sub_para_name)
                                matrix_mask_before = np.reshape(self.mask[sub_para_name],
                                                                tuple(self.para_size[sub_para_name]))
                                matrix_mask_before[:, filter_index, :, :] = 0
                                self.mask[sub_para_name] = np.reshape(matrix_mask_before, -1)

                    elif "addmm" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            if ".weight" in sub_para_name:
                                new_name.add(sub_para_name)
                                matrix_mask_before = np.reshape(self.mask[sub_para_name],
                                                                tuple(self.para_size[sub_para_name]))

                                assert (self.para_size[sub_para_name][1] % self.para_size[para_name][0] == 0)
                                num_features_pre_channel = self.para_size[sub_para_name][1] // self.para_size[para_name][0]
                                logging.debug("In fc, {} input feature map 1 channel of last convolution layer.".format(num_features_pre_channel))
                                # get corresponding feature index
                                real_filter_index = np.array([[idx * num_features_pre_channel+j for j in range(num_features_pre_channel)] for idx in filter_index]).flatten()
                                matrix_mask_before[:, real_filter_index] = 0
                                self.mask[sub_para_name] = np.reshape(matrix_mask_before, -1)

        return new_name

    def __get_filter(self,weight_torch,name):
        filter_norm, filter_index_norm = self.__get_filter_norm2_(weight_torch,name)
        logging.debug("finish built norm filter")
        logging.debug("Zero in norm_filter: {}".format(len(filter_norm)-np.count_nonzero(filter_norm)))
        return filter_norm, filter_index_norm

    def __get_filter_norm2_(self, weight_torch,name):
        assert(len(weight_torch.size()) == 4)
        codebook = self.mask[name]
        filter_pruned_num = int(weight_torch.size()[0] * (1 - self.rate_keep_norm))
        weight_vec = weight_torch.view(weight_torch.size()[0], -1)

        norm = torch.norm(weight_vec, self.norm, 1)
        norm_np = norm.detach().cpu().numpy()
        filter_index = norm_np.argsort()[:filter_pruned_num]
        logging.debug("number of filter pruned as norm{}: {}".format(self.norm, filter_pruned_num))
        logging.debug("norm filter index: {}".format(filter_index))


        kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        for x in range(0, len(filter_index)):
            codebook[filter_index[x] * kernel_length: (filter_index[x] + 1) * kernel_length] = 0

        return codebook,filter_index


# Done
ThiNet_distance = []
tep_output = None
def ThiNet_hook(module, input, output):
    global tep_output
    global ThiNet_distance

    if tep_output is not None:
        ThiNet_distance.append(torch.norm(tep_output - output).item())
    else:
        tep_output = output
class MaskThiNet(Mask):

    def __init__(self, model,dataloader:torch.utils.data.DataLoader,
                 input_shape:tuple,
                 current_epoch=0,
                 use_cuda=True,
                 ratio=0.7,
                 skip_tail_conv_fp: int = 1,
                 skip_tail_conv_cp: int = 1,
                 extra_skip_fp_module_name: list = None,
                 extra_skip_cp_module_name: list = None
                 ):
        Mask.__init__(self,model = model,
                      input_shape=input_shape,
                      skip_tail_conv_fp=skip_tail_conv_fp,
                      skip_tail_conv_cp=skip_tail_conv_cp,
                      extra_skip_cp_module_name=extra_skip_cp_module_name,
                      extra_skip_fp_module_name=extra_skip_fp_module_name,use_cuda=use_cuda)
        self.epoch = -1
        self.current_epoch = current_epoch
        self.current_hook_handle = None
        self.data_loader = dataloader
        self.ratio = ratio
        logging.debug("ratio: {}".format(self.ratio))

    def custom_init(self):
        self.model.eval()

        self.mask_name = []
        for node in self.model_graph.nodes():
            if "convolution" in node.kind():
                module_name = self._scope2name(node.scopeName())
                if module_name in self.skip_cp_module:
                    logging.debug("skip module {}".format(module_name))
                    continue
                logging.debug("put parameter name of module {} in self.mask_name".format(module_name + '.weight'))
                self.mask_name.append(module_name + '.weight')
        logging.debug("Current mask_name is {}".format(self.mask_name))
        self.epoch = len(self.mask_name) * 2 + 12

    @_check_make
    def update_cur_mask_name(self):
        if self.cur_mask_name:
            self.cur_mask_name.clear()
        # get current para
        self.cur_mask_name.append(self.mask_name.pop(0))
        self.mask_name.append(self.cur_mask_name[0])

        logging.debug("update cur_mask to {}".format(self.cur_mask_name))

        for name, module in self.model.named_modules():
            if not list(module.children()):
                if name + ".weight" in self.cur_mask_name:
                    logging.debug("hook the funtion to module {}".format(name))
                    self.current_hook_handle = module.register_forward_hook(ThiNet_hook)


    def update_mask(self):
        global ThiNet_distance
        global tep_output
        for module_name, module in self.model.named_modules():
            if list(module.children()):
                continue
            for para_name, para in module.named_parameters():
                para_name = module_name+"."+para_name
                if para_name in self.cur_mask_name:
                    left_channels = list(range(0, self.para_size[para_name][1]))
                    pruned_channels = []
                    # buff origin parameter
                    ori_para = deepcopy(para.data.cpu())
                    logging.debug("pruning paras name {}".format(para_name))

                    for data, target in self.data_loader:
                        if len(pruned_channels) >= int(self.para_size[para_name][1] * (1 - self.ratio)):
                            break
                        # get origin output by hook
                        if self.use_cuda:
                            data = data.cuda()
                        self.model(data)
                        # prune all pruned channel
                        for c in pruned_channels:
                            para.data[:, c, :, :] = 0
                        # buff pruned_channels parameter
                        pruned_para = deepcopy(para.data.cpu())
                        # prune left channels one by one and record output by hook
                        for c in left_channels:
                            para.data[:, c, :, :] = 0
                            self.model(data)
                            # reset parameter
                            para.data = deepcopy(pruned_para).cuda()
                        torch.cuda.empty_cache()
                        logging.debug("distance number: {}".format(len(ThiNet_distance)))
                        logging.debug("left_channels number: {}".format(len(left_channels)))
                        assert (len(ThiNet_distance) == len(left_channels))
                        logging.debug("distance {}".format(ThiNet_distance))
                        # get minimum reconstruction loss
                        min_index = np.argmin(ThiNet_distance)
                        logging.debug("less error index of current step is {}".format(min_index))
                        pruned_channels.append(left_channels[min_index])
                        logging.debug("pruned_channel {}".format(pruned_channels))
                        left_channels.pop(min_index)
                        logging.debug("left_channel {}".format(left_channels))
                        ThiNet_distance.clear()
                        tep_output = None
                        torch.cuda.empty_cache()
                        # restore parameter
                        para.data = deepcopy(ori_para).cuda()

                    logging.debug("pruned clannels are {}".format(pruned_channels))
                    matrix_mask = np.reshape(self.mask[para_name], tuple(self.para_size[para_name]))
                    for c in pruned_channels:
                        matrix_mask[:, c, :, :] = 0
                    new_mask = np.reshape(matrix_mask, -1)


                    new_name = self.__update_mask(para_name,module_name,new_mask,np.array(pruned_channels))
                    self.cur_mask_name.extend(list(new_name))
        if self.current_hook_handle is not None:
            self.current_hook_handle.remove()
        if ThiNet_distance:
            ThiNet_distance.clear()


    def __update_mask(self,para_name:str,module_name:str,new_mask:np.ndarray,filter_index:np.ndarray) -> set:
        assert(".weight" in para_name)
        new_name = set()
        self.mask[para_name] = new_mask

        # bn, convolution: weight, bias
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) == module_name:
                # find related node for pruning
                related_nodes,_ = self.general_search(node,backward=True)
                logging.debug("pruning channel of {} => pruning filter of {}".format(module_name,[self._scope2name(n.scopeName()) for n in related_nodes]))
                assert (len(related_nodes) > 0)
                for r_node in related_nodes:
                    # bn
                    if "batch_norm" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            new_name.add(sub_para_name)
                            self.mask[sub_para_name][filter_index] = 0
                    # next conv
                    elif "convolution" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            if ".weight" in sub_para_name:
                                new_name.add(sub_para_name)
                                matrix_mask_before = np.reshape(self.mask[sub_para_name],
                                                                tuple(self.para_size[sub_para_name]))
                                matrix_mask_before[filter_index, :, :, :] = 0
                                self.mask[sub_para_name] = np.reshape(matrix_mask_before, -1)
                            if ".bias" in sub_para_name:
                                new_name.add(sub_para_name)
                                self.mask[sub_para_name][filter_index] = 0
        return new_name


    def __objective(self, origin_output, pruned_output):
        return torch.norm(origin_output - pruned_output).item()


# Done
APoZ = []
def APoZ_hook(module, input, output):
    logging.debug("hook input type: {}".format(type(input)))
    logging.debug("hook output type: {}".format(type(output)))

    global APoZ
    num_element =1
    for n in output.size():
        num_element = num_element*n
    # number of element pre layer
    zero_ratio = []
    num_element_pre_channel = num_element / output.size()[1]
    for i in range(output.size()[1]):
        nun_zero_num = torch.nonzero(torch.clamp(output[:,i,:,:],min=0)).shape[0]
        apoz = 1- (nun_zero_num / num_element_pre_channel)
        zero_ratio.append(apoz)
    APoZ.append(zero_ratio)
    logging.debug("Current output size: {}".format(output.size()))
    logging.debug("Current APoZ length: {}".format(len(APoZ)))
class MaskAPoZ(Mask):

    def __init__(self, model,
                 dataloader:torch.utils.data.DataLoader,
                 input_shape:tuple,
                 current_epoch=0,
                 use_cuda=True,
                 ratio=0.7,
                 skip_tail_conv_fp: int = 1,
                 skip_tail_conv_cp: int = 1,
                 extra_skip_fp_module_name: list = None,
                 extra_skip_cp_module_name: list = None
                 ):

        Mask.__init__(self,model = model,
                      input_shape=input_shape,
                      skip_tail_conv_fp=skip_tail_conv_fp,
                      skip_tail_conv_cp=skip_tail_conv_cp,
                      extra_skip_cp_module_name=extra_skip_cp_module_name,
                      extra_skip_fp_module_name=extra_skip_fp_module_name,use_cuda=use_cuda)
        self.epoch = -1
        self.current_epoch = current_epoch
        self.current_hook_handle = None
        self.data_loader = dataloader
        self.ratio = ratio
        logging.debug("ratio: {}".format(self.ratio))

    def custom_init(self):
        self.model.eval()
        # initialize mask_name: only name of conv in mask_name
        self.mask_name = []
        # build relu : [module] map and conv : relu map for convenience
        self.relu_module_name_map = defaultdict(list)
        self.conv_relu_name_map = dict()
        relu_set = set()
        for node in self.model_graph.nodes():
            if "convolution" in node.kind():
                module_name = self._scope2name(node.scopeName())
                if module_name in self.skip_fp_module:
                    logging.debug("skip module {}".format(module_name))
                    continue
                self.target_nodes.append("relu")
                result_list,_ = self.general_search(node)
                self.target_nodes.pop()
                if len([n for n in result_list if "relu" in n.kind()]) == 0:
                    logging.debug("skip module {}".format(module_name))
                    continue
                for relu_node in result_list:
                    if "relu_" in relu_node.kind():
                        # self.relu_conv_name_map[self._scope2name(relu_node.scopeName())].append(self._scope2name(
                        #     node.scopeName()))
                        self.conv_relu_name_map[self._scope2name(node.scopeName())] = \
                            self._scope2name(relu_node.scopeName())
                        relu_set.add(self._scope2name(relu_node.scopeName()))
                logging.debug("put parameter name of module {} in self.mask_name".format(module_name + '.weight'))
                self.mask_name.append(module_name + '.weight')
        logging.debug("Current mask_name is {}".format(self.mask_name))

        # build relu_module_name_map
        for node in self.model_graph.nodes():
            name = self._scope2name(node.scopeName())
            if name in relu_set:
                # input node of relu
                before_node = list(node.inputs())[0].node()
                result_list, valid_list = self.general_search(before_node,backward=True,deep=0)
                # if is ng_node, add it directy
                if not all(valid_list):
                    self.relu_module_name_map[name].append(self._scope2name(before_node.scopeName()))
                # else, find last conv before
                else:
                    conv_list = [node for node in result_list if "convolution" in node.kind()]
                    assert (len(conv_list) == 1, "multiple convolutions before {}: {}".format(name,conv_list))
                    self.relu_module_name_map[name].append(self._scope2name(conv_list[0].scopeName()))


        if len(self.mask_name) == 0:
            logging.error("The model input does not use ReLU activation function or can not find valid node to prune.\nIt cannot be pruned by APoZ.")
            exit(0)
        self.epoch = len(self.mask_name) * 2 + 12


        # for node in self.model_graph.nodes():
        #     if "aten::relu_" in node.kind():
        #         result_list, result_valid = self.general_search(node, backward=True, deep=0)
        #         conv_list = [conv_node for conv_node in result_list if "convolution" in conv_node.kind()]
        #         if all(result_valid) and len(conv_list) == 1:
        #             self.relu_conv_name_map[self._scope2name(conv_list[0].scopeName())] = self._scope2name(node.scopeName())
        #             self.conv_relu_name_map[self._scope2name(node.scopeName())].append(self._scope2name(conv_list[0].scopeName()))
        #             self.relu_list.append(self._scope2name(node.scopeName()))
        #
        #         else:
        #             logging.debug("skip relu module {}".format(self._scope2name(node.scopeName())))

    @_check_make
    def update_cur_mask_name(self):
        if self.cur_mask_name:
            self.cur_mask_name.clear()
        # get current para
        self.cur_mask_name.append(self.mask_name.pop(0))
        self.mask_name.append(self.cur_mask_name[0])
        logging.debug("update cur_mask to {}".format(self.cur_mask_name))
        # get conv module name
        conv_module_name = ".".join(self.cur_mask_name[0].split(".")[:-1])
        logging.debug("conv_module_name: {}".format(conv_module_name))
        logging.debug("conv_relu_name_map: {}".format(self.conv_relu_name_map))
        logging.debug("relu_module_map: {}".format(self.relu_module_name_map))



        assert (conv_module_name in list(self.conv_relu_name_map.keys()),"{} is not in key of conv_relu map: {}".format(
            conv_module_name,list(self.conv_relu_name_map.keys())))

        for name, module in self.model.named_modules():
            # get corresponding relu-module name and hook
            if name == self.conv_relu_name_map[conv_module_name]:
                logging.debug("hook the funtion to module {}".format(name))
                self.current_hook_handle = module.register_forward_hook(APoZ_hook)
                break

    @_check_make
    def update_mask(self):
        torch.cuda.empty_cache()
        for data, target in self.data_loader:
            model = self.model
            if self.use_cuda:
                data = data.cuda()
                model = model.cuda()
            model.eval()
            model(data)
            break

        extra_name = set()
        for module_name, module in self.model.named_modules():
            if not list(module.children()):
                for para_name, para in module.named_parameters():
                    para_name = module_name+"."+para_name
                    if para_name in self.cur_mask_name:
                        new_mask, filter_index = self.__get_filter(module_name,para_name)
                        # update mask of weight, bias, bn and next convolution
                        extra_name.update(self.__update_mask(para_name,module_name,new_mask,filter_index))

        self.cur_mask_name.extend(list(extra_name))

        global APoZ
        self.current_hook_handle.remove()
        if len(APoZ) != 0:
            APoZ.clear()
        torch.cuda.empty_cache()

    # update mask of weight, bias, bn and next convolution/fc
    def __update_mask(self,para_name:str,module_name:str,new_mask:np.ndarray,filter_index:np.ndarray) -> set:
        assert(".weight" in para_name)
        new_name = set()

        # weight
        self.mask[para_name] = new_mask
        bias_name = para_name.replace(".weight",".bias")
        # bias
        if bias_name in self.moduels_para_map[module_name]:
            new_name.add(bias_name)
            self.mask[bias_name][filter_index] = 0

        # bn, next_conv/fc
        for node in self.model_graph.nodes():
            if "convolution" in node.kind() and self._scope2name(node.scopeName()) == module_name:
                related_nodes,related_valid= self.general_search(node)
                logging.debug("pruning filter of {} => pruning channel of {}".format(module_name,[self._scope2name(n.scopeName()) for n in related_nodes]))
                assert (len(related_nodes) > 0 and all(related_valid))
                for r_node in related_nodes:
                    # bn
                    if "batch_norm" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            new_name.add(sub_para_name)
                            self.mask[sub_para_name][filter_index] = 0
                    # next conv
                    elif "convolution" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            if ".weight" in sub_para_name:
                                new_name.add(sub_para_name)
                                matrix_mask_before = np.reshape(self.mask[sub_para_name],
                                                                tuple(self.para_size[sub_para_name]))
                                matrix_mask_before[:, filter_index, :, :] = 0
                                self.mask[sub_para_name] = np.reshape(matrix_mask_before, -1)

                    # TODO:reshape / view fc
                    elif "addmm" in r_node.kind():
                        for sub_para_name in self.moduels_para_map[self._scope2name(r_node.scopeName())]:
                            if ".weight" in sub_para_name:
                                new_name.add(sub_para_name)
                                matrix_mask_before = np.reshape(self.mask[sub_para_name],
                                                                tuple(self.para_size[sub_para_name]))

                                assert (self.para_size[sub_para_name][1] % self.para_size[para_name][0] == 0)
                                num_features_pre_channel = self.para_size[sub_para_name][1] // self.para_size[para_name][0]
                                logging.debug("In fc, {} input feature map 1 channel of last convolution layer.".format(num_features_pre_channel))
                                # get corresponding feature index
                                real_filter_index = np.array([[idx * num_features_pre_channel+j for j in range(num_features_pre_channel)] for idx in filter_index]).flatten()
                                matrix_mask_before[:, real_filter_index] = 0
                                self.mask[sub_para_name] = np.reshape(matrix_mask_before, -1)

        return new_name

    # get filter index and new mask by APoZ list of current convolution node
    def __get_filter(self,module_name,para_name):

        relu_name = self.conv_relu_name_map[module_name]
        module_list = self.relu_module_name_map[relu_name]
        try:
            conv_idx = module_list.index(module_name)
        except ValueError:
            raise ValueError("Convolution {} is not in list of {}:{}.".format(module_name,relu_name,module_list))

        pruned_index = np.array(APoZ[conv_idx]).argsort()[::-1][:int(len(APoZ[conv_idx]) * (1-self.ratio))]

        print("para_size = {}".format(self.para_size[para_name]))
        kernel_length = self.para_size[para_name][1] * self.para_size[para_name][2] * self.para_size[para_name][3]
        codebook = self.mask[para_name]
        for x in range(0, len(pruned_index)):
            codebook[pruned_index[x] * kernel_length: (pruned_index[x] + 1) * kernel_length] = 0
        return codebook, pruned_index









class MaskRandom(Mask):
    def __init__(self, model):
        Mask.__init__(self, model)


class MaskOracel(Mask):
    def __init__(self, model):
        Mask.__init__(self, model)


class MaskDCP(Mask):
    def __init__(self, model):
        Mask.__init__(self, model)

    @_check_make
    def get_filter(self,weight_torch,length):
        pass

    def init_rate(self):
        pass


class MaskNISP(Mask):
    def __init__(self, model):
        Mask.__init__(self, model)


class MaskACM(Mask):
    def __init__(self, model):
        Mask.__init__(self, model)


class MaskApproxOrecel(Mask):
    def __init__(self, model):
        Mask.__init__(self, model)





