# Pruning torch 开发说明

这篇readme作为开发指南,主要是记录Mask模块的一些项目结构,实现思路和代码细节,并会记录一下当前开发进度.



 
## TODO List:
 - [x] remake function after hard_pruning
 - [ ] test remake function
 - [ ] Split MaskImpt
 - [ ] ModuleList Support (以支持YOLOv3的剪枝为标注)
 - [ ] (average_pool + ) flatten + linear support
 - [ ] pre-layer intern-pruning finetune [callback function]
 - [ ] demo code with fineturn
 - [ ] setup as module
 - [ ] multi-GPU fineturn

## 项目结构和思路

项目结构很简单,只有2层,Mask类为所有pruning实现类的基类,实现了底层的torch模型解析,往前往后的条件结构搜索,基于Mask的软剪枝,以及对权重和参数直接修改的硬剪枝等. Mask类还定义了一个通用的剪枝执行顺序和框架流程,当开发者实现新pruning算法时,只需要集中精力实现pruning的逻辑即可.

具体的说,一个pruning算法对应一个Mask基类的子类,并根据既定的流程和要求,通过覆盖实现指定的几个成员方法，通过更新Mask基类的几个关键属性来实现剪枝算法即可,无需关心剪枝的实际操作（即model surgery的过程已经实现了，开发者只需要给'说明书'即可）.

Mask基类中有几个关键的属性是开发新purning算法时需要用到并更新的，开发者通过实现自己的算法那来更新这几个关键的属性，Mask基类会使用这几个属性对模型进行剪枝。

所谓Mask，其实就是掩膜的概念，在所有可以进行剪枝的模型权重上覆盖上一层shape相同的0/1 mask，每轮剪枝只需更新mask的值即可，Mask基类会根据维护的mask进行软剪枝和硬剪枝操作。

* 软剪枝：权重*当前mask，将对应权重归0，这种方法只能获得剪枝后对应的结果，没有降低模型的计算量和参数量

* 硬剪枝：根据当前mask的情况，对模型权重（当前包括，卷积权重，bias，全连接权重，BN的mean和var）做修改，直接修改其shape和值，剪枝后模型参数量和计算量下降，模型输出需要保证和软剪枝后的模型相同

Note:

1. 关于软剪枝以后的训练，根据后向传播链式法则推导,当某个'乘法'权重被置0后,由于其输出也是0,则对loss的梯度恒为0,神经元死亡,接下来的训练中不会被更新. 
2. 但是如果卷积有bias项,即使在软剪枝后将bias置0,若不进行硬剪枝,在接下来的fineturn过程中,bias还是会根据梯度更新,很明显这不符合我们的期望
3. 对于2,模型中有batchnorm的情况同理.

因此需要注意: 当模型的算子有bias时,retrain(fineturn)步骤需要在执行硬剪枝以后进行,如果确保没有bias,则无所谓.



## pruning的流程

1. 初始化Mask类的子类(make)
    * 初始化Mask基类：定义成员属性
    * 实现类成员初始化，包括自定义初始化方法：调用make方法，make会利用torch.jit模块解析模型结构并初始化各个属性，最后会调用'可选择覆盖的自定义初始化方法（custom_init方法）'。不同的剪枝算法有各自需要维护的属性，可以通过对基类的custom_init方法的覆盖实现初始化。

2. soft-pruning阶段(domask)
    * 在每轮的pruning之前，一般需要对几个关键的属性做更新：开发者需要实现update_mask_name，update_cur_mask_name，update_mask三个方法，来更新当前的mask（mask是对应模型中所有权重[named_parameter]的0/1 code book.注意,因为某些设计原因,mask是一维的）
    * 根据更新后本轮的mask做软剪枝,对权重置0

3. hard-pruning(generate_pruned_model)
    * 根据当前mask对模型做剪枝,切割的部分包括卷积的weight和bias,BN的γ,β以及runningmean,runningvar,fc的weight和bias.

4. 获得hard-pruning后模型,进行fineturn
    

## 基类Mask说明

这里挑选几个比较重要的方法和属性做说明

1. Mask类初始化

    通用的初始化的实现在core/MaskPrune.py中Mask基类中：
    * Mask.__init__方法的输入：
        * model: torch.nn.Module, 需要做剪枝的模型
        * input_shape: tuple, 输入的大小,(B,C,H,W)
        * extra_skip_fp_module_name: list = None, 额外在filter pruning中被skip的module名(filter不会被置0)
        * extra_skip_cp_module_name: list = None, 额外在channel pruning中被skip的module名(channel不会被置0)
        * skip_tail_conv_fp: int = 1,不会被fp的尾部conv模块深度,从后往前数
        * skip_tail_conv_cp: int = 1,不会被cp的尾部conv模块深度,从后往前数
        * use_cuda: bool = True, 是否用cuda
    
    * self.make()
    
        主要用于解析模型结构,初始化构建关键成员变量等工作,并会在最后调用custom_init方法,可以定制覆盖初始化函数custom_init来进行自定义初始化.

    不考虑custom_init方法,调用make后,对象会有以下成员变量,可供剪枝算法调用,比较重要的几个成员变量单独说明:

    * self.model_graph: 
    
        torch._C.Graph对象,通过torch.jit.trace(self.model).graph获得,是实现模型解析和剪枝的关键属性.
        在实现pruning算法时,该属性的一般用法:
        1. self.model_graph.nodes()获得所有graph下的torch._C.Node对象
        2. 用Node.kind()方法获得Node的动作类型(string类型),并过滤出想要的Node(比如'convolution'in Node.kind(),这个需要自己做实验确定想要的模块返回string具体是什么)
        3. 通过general_search方法前向或后向搜索到想要的与当前Node关联的其他Node
        3. 通过self._scope2name(node.scopeName())获得该Node对应的ModuleName
        4. 现在,可以通过Mask类维护的各种dictionary获取对应的Module和parameter了

    * self.skip_fp_module: 
    
        list,所有需要在fp过程中被skip的module的ModuleName,包括初始化时的extra_skip_fp_module_name,是实现模型解析和剪枝的关键属性.

    * self.skip_cp_module: 
        
        list,所有需要在cp过程中被skip的module的ModuleName,包括初始化时的extra_skip_cp_module_name,是实现模型解析和剪枝的关键属性.
    
    * self.mask: 
        
        dict,初始化的mask字典,parameter_name:np.ones(para_length),Mask类的核心属性.注意,每个mask是一维的np.ndarray向量,初始化为全1向量.
        在实现pruning算法时,一般会在覆盖的self.update_mask方法中更新.在do_mask中,会根据当前的self.mask对所有parameter做软剪枝的操作.
    
    * self.mask_name: 
        
        list, 默认初始化的mask_name是所有named_parameter的list,该属性是用来维护所有需要剪枝的module的parameter对对应在self.mask中的key,
        一般会在custom_init方法中重新被初始化(比如初始化成只包含需要剪枝的parameter name),在pruning算法实现中,在self.update_mask_name方法中更新.
    
    * self.cur_mask_name: 
        
        list,初始化的cur_mask_name是所有named_parameter的list,同self.mask_name,该属性是用来维护下一轮do_mask方法调用时,需要剪枝的module的parameter对对应在self.mask中的key,
        一般会在custom_init方法中重新被初始化(比如初始化成只包含需要剪枝的parameter name),在pruning算法实现中,在self.update_cur_mask_name方法中更新.
    
    * 其他
        * self.model: init方法传进来的module
        * self.para_size: dict,模型所有parameter的Tensor.size,在更新mask时提供便利
        * self.para_length: dict,模型所有parameter的元素个数,在更新mask时提供便利
        * self.moduels_para_map: dict,模型所有叶子module(没有children的module)对应其parameter list,在找Parameter时提供便利
        * self.skip_tail_conv_fp: 初始化时的skip_tail_conv_fp
        * self.skip_tail_conv_cp: 初始化时的skip_tail_conv_cp
        * self.extra_skip_cp_module_name: 初始化时的extra_skip_p_module_name
        * self.extra_skip_fp_module_name: 初始化时的extra_skip_fp_module_name

2. do_mask方法
    
    实现模型的软剪枝,执行顺序是: update_mask_name -> update_cur_mask_name -> update_mask -> do_pruning
    
    由于bias的原因,软剪枝以后的模型不能直接训练,需要经过硬剪枝才能进行fineturn.
    
    update_mask_name; update_cur_mask_name; update_mask 三个方法需要开发者根据pruning算法更新self.mask, self.mask_name和self.cur_mask_name,在子类中实现

3. generate_pruned_model方法
    
    该方法不需要开发者维护.该方法是在软件剪枝完成以后,对模型进行硬剪枝,剪枝后的模型实际参数量减少,输出接软剪枝后模型保持一致.
    剪枝完后会调用self.__remake()从新构建Mask类中的几个属性,并在最后调用在子类中可选覆盖的custom_remake方法做重新构建:
    
    * self.model
    * self.mask
    * self.para_length
    * self.para_size
    
    self.mask_name和self.cur_mask_name不会被修改,开发者可以自己维护这两个变量,控制剪枝的pipeline和顺序
    
    方法返回剪枝后的model(self.model)

4. general_search方法

    该方法不需要开发者维护.该方法可以往前或往后迭代搜索指定目标的module对应的torch._C.Node对象,该方法可以很方便的对torch模型进行深度优先的搜索,
    并返回路径上的目标节点对象,同时可以设置减深度节点,合法终止点和非法终止点 (self.target_nodes, self.valid_stop_nodes, self.valid_stop_nodes, self.ng_nodes)
    以此控制搜索过程.
    
    方法返回[torch._C.Node对象],[boolean],第一个list为所有路径上的self.target_nodes, self.valid_stop_nodes, self.valid_stop_nodes,第二个list中如果出现False,则搜索过程中遇到ng_nodes.
    一般当遇到ng_node,则当前节点不能被剪枝.

5. 几个子类需要根据实际算法覆盖的成员方法:
    * custom_init: 自定义初始化,在make方法最后调用
    * custom_skip_module_names: 自定义module_skip,结果会加入到self.skip_fp_module和self.skip_cp_module中
    * update_mask_name: do_mask方法第一步调用,用于更新self.mask_name
    * update_cur_mask_name: do_mask方法第一步调用,用于更新self.cur_mask_name
    * update_mask: do_mask方法第一步调用,用于更新self.mask
    * custom_remake: self.__remake方法中最后调用,用于自定义硬剪枝后的重构步骤

    
## 待解决的关键问题

1. generate_pruned_model进行硬剪枝后,没有继续维护类成员变量,导致当前的实现的pruning是一次性的,既调用完generate_pruned_model后类对象便无法复用了.
   如果需要实现:partial pruning (for example, pruned a filter)->fineturn -> partial pruning (another filter),则需要从新构建新的Mask对象,并且将已经剪枝的filter加入extra_skip_fp_module_name或extra_skip_cp_module_name中,对多段的剪枝/fineturn的算法不友好:
    *  需要实现一个general的remake方法,在generate_pruned_model后更新和维护成员变量
2. ModuleList的支持: 由于torch.jit的模型解析bug,如果Module中使用了ModuleList,则模型解析会出现域名重复和错乱的问题.
    *  经过实验,一个解决办法是重构网络Module对象,如果发现当前Module中包含ModuleList:
        
        将ModuleList中的Module取出来,并作为当前module的子module属性赋予成员变量名
    *  经过上面的重构,jit可以解析出正确的模型结构,获得对应Node的名字.实验可以参考test_ModelParse.py代码
3. 需要实现Mask类的callback function支持,用callback的方式管理fineturn流程,不需要来回从Mask中取module.

4. 支持average_pool和flatten后加linear的结构(当前设置为ng_node)


## 如何用Mask类实现pruning算法

已经实现的几个Pruning算法在MaskImpt/MaskImpt.py中,可以用作参考,可以从简单的实现看起.
这里以MaskImpt.py中实现的MaskFilterNorm子类来作为例子说明.

### 算法说明

MaskFilterNorm就是默认卷积核的norm-2越小,重要性越低,可以被剪枝.那么我们要做的就是统计所有可以被剪枝的卷积的每个卷积核的norm-2,然后选择每层最小的几个卷积核做剪枝.

### 如何实现

1. 构造函数继承Mask类,并扩展维护额外三个成员变量self.rate_keep_norm, self.pre_layer和self.norm
2. 覆盖实现custom_init,重新初始化self.mask_name和self.cur_mask_name.只考虑'不在self.skip_fp_module中的,可以进行filter pruning的卷基层的weight"
    ```python
    
    def custom_init(self):
        self.mask_name = []
        # 遍历所有node
        for node in self.model_graph.nodes():
            # 选出卷基层
            if "convolution" in node.kind():
                # 根据node.scopeName转换出卷积module的名字
                module_name = self.scope2name(node.scopeName())
                #过滤掉所有无法进行fp的层
                if module_name in self.skip_fp_module:
                    logging.debug("skip module {}".format(module_name))
                    continue
                logging.debug("put parameter name of module {} in self.mask_name".format(module_name+'.weight'))
                # 将module_name.weight放进mask_name, 这里放进去的name和Module.named_parameter()或得到的name是一样的
                self.mask_name.append(module_name+'.weight')
        logging.debug("Current mask_name is {}".format(self.mask_name))
        self.cur_mask_name = deepcopy(self.mask_name)
    
    ```
3. 开始实现剪枝算法部分,我们要做的是根据算法的逻辑更新cur_mask_name和self.mask,剪枝的操作只需调用domask()即可.
    * update_cur_mask_name
    
        注意要加上_check_make装饰器.如果不是pre_layer的方法,则self.cur_mask_name就是self.mask_name,既一次全部剪完.
        如果是pre_layer,则每次剪一层,做法是先清空当前cur_mask_name,然后从mask_name中每次按顺序取一个放到cur_mask_name
        
    * update_mask
    
        然后到覆盖update_mask方法,这个方法中需要根据cur_mask_name中mask_name更新self.mask中的0/1 code-book.这里说明的是一次性剪枝,不是pre-layer
        在custom_init中,我们将所有可剪枝的卷积的weight的名字放进cur_mask_name.此时要做的:
        1. 根据这些name获取卷积核的weight(.data),并计算norm,根据norm的大小,决定将对应mask中的部分改成0,生成新mask(self.__get_filter方法)
        2. 根据返回的filter_index,修改self.mask中的值(self.__update_mask)
           除了卷基层的的weight的mask本身之外,还需要处理的几个mask:
            * 该层卷积的bias的mask
            * 下一层卷积的每个卷积核的对应channel的mask
            * 如果有BN层,则还需处理bn的mask
            * (待支持) 后接GAP+flatten+linear或直接flatten+linear
        3. 把之前不在cur_mask_name中的mask_name[比如bias,bn的mask等]加入进去

这样,一个pruning算法就实现好了.



        



