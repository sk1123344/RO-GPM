import os
import numpy as np
import pandas as pd
import collections
import copy
import torch
from Model import ResNet18Fundus
import pynvml


def _get_dataset_order(self, sub_num=None):
    order_list = []
    if sub_num is None:
        for set_path in self.args.dataset_path[:-1]:
            order_list.append(os.path.basename(set_path))
    elif sub_num == -1:
        for set_path in self.args.dataset_path:
            order_list.append(os.path.basename(set_path))
    else:
        for set_path in self.args.dataset_path[:sub_num]:
            order_list.append(os.path.basename(set_path))
    return '-'.join(order_list)


def occupy_memory(cuda_device, mem_requirement):
    """ Create a large tensor and delete it.
    This operation occupies the GPU memory, so other processes cannot use the occupied memory.
    It is used to ensure that this process won't be stopped when it requires additional GPU memory.
    Be careful with this operation. It will influence other people when you are sharing GPUs with others.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(cuda_device))
    tensor_list = []
    current_occupy_mem = 0
    print('=occupy mem=')
    while current_occupy_mem < mem_requirement:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # print('free = {}, used = {}'.format(mem_info.free / 1024 ** 2, mem_info.used / 1024 ** 2))
        current_available_mem = mem_info.free / 1024 ** 2 - 1100
        current_require = mem_requirement - current_occupy_mem
        block_mem = int(current_available_mem if current_available_mem < current_require else current_require)
        if block_mem > 0:
            tensor_list.append(torch.FloatTensor(256, 1024, block_mem).to('cuda'))
            current_occupy_mem += block_mem
    if tensor_list:
        for idx in range(len(tensor_list)):
            del tensor_list[idx]


def set_gpu(x):
    """ Set up which GPU we use for this process """
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('Using gpu:', x)


def dict_append(dict_, column_name_, data_):
    if column_name_ not in dict_.keys():
        dict_[column_name_] = [data_] if not isinstance(data_, list) else data_
    else:
        dict_[column_name_].append(data_) if not isinstance(data_, list) else dict_[column_name_].extend(data_)
    return dict_


def check_and_make_dir(dir_):
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    return dir_


def dict_save(dict_, name_, path_=None):
    pd.DataFrame(dict_).to_csv(os.path.join(os.getcwd() if path_ is None else path_, '{}.csv'.format(name_)),
                               index=False)


def acc_matrix_to_dict(acc_matrix_):
    """

    :param acc_matrix_: square
    :return:
    """
    dict_ = collections.OrderedDict()
    for i_ in range(acc_matrix_.shape[1]):
        dict_append(dict_, 'task{}'.format(str(i_)), list(acc_matrix_[:, i_]))
    return dict_


def acc_matrix_print(acc_matrix_, phase_id_=None):
    for i in range(acc_matrix_.shape[0] if phase_id_ is None else (phase_id_ + 1)):
        for j in range(acc_matrix_.shape[0] if phase_id_ is None else (phase_id_ + 1)):
            print('{:.4f}\t'.format(acc_matrix_[i, j]), end='')
        print('\n')


def dict_padding(dict_):
    max_list_length = 0
    _dict_ = copy.deepcopy(dict_)
    for k_, v_ in _dict_.items():
        if not isinstance(v_, list):
            v_ = [v_]
            _dict_[k_] = v_
        if len(v_) > max_list_length:
            max_list_length = len(v_)
    for k_, v_ in _dict_.items():
        if not isinstance(v_, list):
            v_ = [v_]
            _dict_[k_] = v_
        pad_size = max_list_length - len(v_)
        _dict_[k_].extend([None] * pad_size)
    return _dict_


def model_selection(dataset_name_, base_nc_, network_architecture_, use_pretrained_backbone_):
    if dataset_name_.lower() == 'fundus':
        if network_architecture_ == 'resnet18':
            return ResNet18Fundus(base_nc_, use_pretrained_backbone=use_pretrained_backbone_)


def forgetting_measure(acc_matrix_):
    return np.mean(acc_matrix_[-1, :-1] - np.diag(acc_matrix_)[:-1])


def fwt_measure(acc_matrix_):
    return np.mean(np.diag(acc_matrix_)[1:] - acc_matrix_[0, 1:])


def _check_resized_dataset(path_, resize_shape):
    if isinstance(path_, str):
        if os.path.exists('{}_{}'.format(path_, resize_shape)):
            return '{}_{}'.format(path_, resize_shape)
        else:
            return path_
    elif isinstance(path_, list):
        re_list = []
        for p_ in path_:
            if os.path.exists('{}_{}'.format(p_, resize_shape)):
                re_list.append('{}_{}'.format(p_, resize_shape))
            else:
                re_list.append(p_)
        return re_list
