from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data.sampler import Sampler
import numpy as np
from sklearn.utils import shuffle
import copy
import os
from PIL import Image
import random
import skimage.io
import urllib.request
import pickle
from tqdm import tqdm
import torch.nn as nn


class SessionSampler(Sampler):
    """
    use np.random.seed to reproduce the sampling results of GPM
    """
    def __init__(self, current_data_length_, random_seed_, shuffle_=True):
        super(SessionSampler, self).__init__(None)
        self.random_seed_ = random_seed_
        self.shuffle = shuffle_
        self.r = np.arange(current_data_length_)
        if self.shuffle:
            np.random.seed(random_seed_)
            np.random.shuffle(self.r)
        self.sequence_list = list(self.r)

    def sample_list_update(self):
        if self.shuffle:
            np.random.seed(self.random_seed_)
            np.random.shuffle(self.r)
            self.sequence_list = list(self.r)

    def __iter__(self):
        return iter(self.sequence_list)

    def __len__(self):
        return 1


def image_load_preprocess(image_path_, transform_):
    try:
        image = Image.fromarray(skimage.io.imread(image_path_)).convert('RGB')
    except Exception as e:
        print('wrong path: {}'.format(image_path_))
        image = np.ones((84, 84, 3), dtype=float)
    return transform_(image) if transform_ is not None else image


def l2_distance_calculation(tensor_a_, tensor_b_):
    """
    calculate for each row
    :param tensor_a_: [n, dim]
    :param tensor_b_:
    :return: n
    """
    assert tensor_b_.size() == tensor_a_.size(), print('different size for tensor_a {}, tensor_b {}'.format(tensor_a_.size(), tensor_b_.size()))
    return torch.sqrt(torch.sum((tensor_a_ - tensor_b_) ** 2, dim=1))


def _herding_process(features_, sample_num_, herding_method):
    """

    :param features_:
    :return:
    """
    prototype = torch.mean(features_, dim=0)
    if herding_method == 'l2':
        distance_ = l2_distance_calculation(features_, prototype.repeat((features_.size(0), 1)))
    elif herding_method == 'cos':
        distance_ = nn.CosineSimilarity(dim=1)(features_, prototype.repeat((features_.size(0), 1)))
    else:
        raise ValueError('herding method {}'.format(herding_method))
    return torch.sort(distance_, dim=0, descending=False)[1].cpu().numpy()[: sample_num_]


class InfiniteBatchSampler(Sampler):
    """
    used for replay, which can sample from a small buffer according to current phase's sample num and batch size and need not to reinitialize
    the dataloader every iteration
    """
    def __init__(self, buffer_size, epoch_sample_size, batch_size, drop_last=False):
        super(InfiniteBatchSampler, self).__init__(None)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError('batch size = {}'.format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError('drop last = {}'.format(drop_last))
        self.buffer_size = buffer_size
        self.epoch_sample_size = epoch_sample_size
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        total_iter = self.__len__()
        for idx in range(total_iter):
            if idx == total_iter - 1:
                batch_size = self.epoch_sample_size % self.batch_size
            else:
                batch_size = self.batch_size
            yield list(np.random.choice(range(self.buffer_size), size=batch_size, replace=False if batch_size <= self.buffer_size else True))

    def __len__(self):
        if self.drop_last:
            return self.epoch_sample_size // self.batch_size
        else:
            return (self.epoch_sample_size + self.batch_size - 1) // self.batch_size


class TempDataset(Dataset):
    def __init__(self, transform_, image_data_, label_data_, load_to_ram=False):
        self.image = image_data_
        self.label = label_data_
        self.transform_ = transform_
        self.load_to_ram = load_to_ram

    def __getitem__(self, item):
        return self.transform_(self.image[item]) if self.load_to_ram else image_load_preprocess(self.image[item], self.transform_), self.label[item]

    def __len__(self):
        return self.label.size(0)


class BaseDataset(Dataset):
    def __init__(self, train_transform_, test_transform_, load_to_ram_=True, random_order=False, validation_percentage=0, random_seed=1993):
        """
        data_dict{'train': {'phase_num': {'x': np.array, 'y': torch.tensor}, ...}, ...}
        up_to_now/joint_data{'train': {'x': np.array, 'y': torch.tensor}, ...}
        :param train_transform_:
        :param test_transform_:
        :param load_to_ram_:
        :param random_order:
        :param validation_percentage:
        :param random_seed:
        """
        self.current_phase = 0

        self.current_set = 'train'
        self.mode = {'train': 'current_phase', 'test': 'current_phase', 'val': 'current_phase'}
        self.use_validation = True if 0 < validation_percentage < 1 else False
        assert 0 <= validation_percentage < 1, print('invalid validation percentage {}'.format(validation_percentage))
        self.validation_percentage_ = validation_percentage
        self.random_seed = random_seed
        self._set_random_seed()
        self.transform_dict = {
            'train': train_transform_,
            'val': test_transform_,
            'test': test_transform_
        }
        self.task_num = None
        self.total_nc = None
        self.data_dict = {}
        self.up_to_now_data = {'train': {}, 'val': {}, 'test': {}}
        self.joint_data = {'train': {}, 'val': {}, 'test': {}}
        self.load_to_ram = load_to_ram_
        self.random_order = random_order
        self.training_set_transform = True
        self.linear_buffer = {'x': None, 'y': None}
        self.memory_dict = {'x': [], 'y': [], 'total_capacity': 0, 'current_use': 0, 'current_use_for_each_class': {}}

    def total_phase_obtain(self):
        return self.task_num

    def _set_random_seed(self):
        seed_ = self.random_seed
        random.seed(seed_)
        np.random.seed(seed_)
        torch.manual_seed(seed_)
        torch.cuda.manual_seed(seed_)
        torch.cuda.manual_seed_all(seed_)

    def data_dict_obtain_(self):
        raise NotImplementedError('data dict not implemented')

    def memory_update(self):
        raise NotImplementedError('memory update not implemented')

    def gather_up_to_now_data(self):
        self.up_to_now_data[self.current_set] = {'x': None, 'y': None}
        for phase in range(self.current_phase + 1):
            if self.up_to_now_data[self.current_set]['x'] is None:
                self.up_to_now_data[self.current_set]['x'] = copy.deepcopy(self.data_dict[self.current_set][phase]['x'])
                self.up_to_now_data[self.current_set]['y'] = self.data_dict[self.current_set][phase]['y'].clone().detach()
            else:
                self.up_to_now_data[self.current_set]['x'] = np.concatenate([self.up_to_now_data[self.current_set]['x'], self.data_dict[self.current_set][phase]['x']], axis=0)
                self.up_to_now_data[self.current_set]['y'] = torch.cat([self.up_to_now_data[self.current_set]['y'], self.data_dict[self.current_set][phase]['y']], dim=0)

    def gather_joint_data(self):
        self.joint_data[self.current_set] = {'x': None, 'y': None}
        for phase in range(self.total_phase_obtain()):
            if self.joint_data[self.current_set]['x'] is None:
                self.joint_data[self.current_set]['x'] = copy.deepcopy(self.data_dict[self.current_set][phase]['x'])
                self.joint_data[self.current_set]['y'] = self.data_dict[self.current_set][phase]['y'].clone().detach()
            else:
                self.joint_data[self.current_set]['x'] = np.concatenate([self.joint_data[self.current_set]['x'], self.data_dict[self.current_set][phase]['x']], axis=0)
                self.joint_data[self.current_set]['y'] = torch.cat([self.joint_data[self.current_set]['y'], self.data_dict[self.current_set][phase]['y']], dim=0)

    def linear_memory_update(self, sample_, label_):
        if self.linear_buffer['x'] is None:
            self.linear_buffer['x'] = sample_.detach().cpu()
            self.linear_buffer['y'] = label_.detach().cpu()
        else:
            self.linear_buffer['x'] = torch.cat([self.linear_buffer['x'], sample_.detach().cpu()], dim=0)
            self.linear_buffer['y'] = torch.cat([self.linear_buffer['y'], label_.detach().cpu()], dim=0)

    def check_if_use_validation(self):
        return self.use_validation

    def set_current_phase(self, phase_):
        assert 0 <= phase_ < self.task_num
        self.current_phase = phase_

    def reset_mode(self, mode_, s_name_):
        assert mode_ in ('up_to_now', 'current_phase', 'joint')
        assert s_name_ in self.mode.keys()
        self.mode[s_name_] = mode_

    def reset_random_seed(self, seed_):
        self.random_seed = seed_

    def reset_set_name(self, set_name_):
        """

        :param set_name_:
        :return:
        """
        assert set_name_ in self.data_dict.keys()
        self.current_set = set_name_

    def reset_all_and_prepare_data(self, s_name_, mode_, phase_):
        self.reset_set_name(s_name_)
        self.reset_mode(mode_, s_name_)
        self.set_current_phase(phase_)
        if mode_ == 'joint':
            self.gather_joint_data()
        if mode_ == 'up_to_now':
            self.gather_up_to_now_data()
        return

    def _get_image_array(self, image, selection=None):
        image_all = None
        if self.load_to_ram:
            return image if selection is None else image[selection]
        else:
            if selection is not None:
                assert len(selection) == len(image), print('different length of selection and image_list')
            for idx, p_ in enumerate(image):
                if selection is not None:
                    if not selection[idx]:
                        continue
                if image_all is None:
                    image_all = np.expand_dims(np.array(image_load_preprocess(p_, None)), axis=0)
                else:
                    image_all = np.concatenate([image_all, np.expand_dims(np.array(image_load_preprocess(p_, None)), axis=0)], axis=0)
        return image_all

    def _get_feature(self, model, image, device_, transform_=None):
        with torch.no_grad():
            image_all = None
            # TODO: ADD a dataloader here to increase speed
            for idx in range(image.shape[0]):
                if image_all is None:
                    image_all = transform_(image[idx]).unsqueeze(0)
                else:
                    image_all = torch.cat([image_all, transform_(image[idx]).unsqueeze(0)], dim=0)
            model.to(device_)
            feature_all = model.feature_extract(image_all.to(device_))
            return feature_all, image_all

    def origin_sample_selection(self, num_dict_: dict, phase_: int, select_sample_per_class_: bool, sample_method_: str = 'random'):
        """
        only select origin sample (path or tensor)
        """
        re_list = []
        re_list_label = []
        np.random.seed(self.random_seed)
        for k, v in num_dict_.items():
            if k not in self.data_dict.keys():
                continue
            transform_action = transforms.Compose([transforms.ToTensor()])
            if select_sample_per_class_:
                for label_ in torch.unique(self.data_dict[k][phase_]['y']):
                    if sample_method_ == 'random':
                        index_ = list(self.data_dict[k][phase_]['y'] == label_)
                        sample_list = []
                        for i_, is_true in enumerate(index_):
                            if is_true:
                                sample_list.append(self.data_dict[k][phase_]['x'][i_])
                        r = np.arange(len(sample_list))
                        np.random.shuffle(r)
                        r = list(r[:v])
                        re_list.extend([sample_list[x] for x in r])
                    else:
                        raise ValueError('invalid sample method {}'.format(sample_method_))
                    # r = list(r[:v])
                    # re_list.extend([self.transform_dict['test'](label_sample[x]).unsqueeze(0) for x in r])
                    # print(len(re_list))
                    re_list_label.extend([label_.unsqueeze(0)] * len(r))
            else:
                r = np.arange(len(self.data_dict[k][phase_]['y']))
                np.random.shuffle(r)
                r = list(r[:v])
                re_list.extend([self.data_dict[k][phase_]['x'][x] for x in r])
                re_list_label.extend([self.data_dict[k][phase_]['y'][x].clone().detach().unsqueeze(0) for x in r])
        return re_list, torch.cat(re_list_label, dim=0)

    def raw_sample_selection(self, num_dict_: dict, phase_: int, select_sample_per_class_: bool, sample_method_: str = 'random', model_=None, device_='cpu', herding_method='l2'):
        """
                select samples for svd
                herding here simply use top-k features closest to the prototype
                if herding is used, select sample for each class
                :param herding_method: l2/cos
                :param device_:
                :param model_:
                :param select_sample_per_class_:
                :param phase_:
                :param sample_method_: ('random', 'herding')
                :param num_dict_:
                :return: sample tensor
                """
        re_list = []
        re_list_label = []
        np.random.seed(self.random_seed)
        for k, v in num_dict_.items():
            if k not in self.data_dict.keys():
                continue
            transform_action = transforms.Compose([transforms.ToTensor()])
            if select_sample_per_class_:
                for label_ in torch.unique(self.data_dict[k][phase_]['y']):
                    if sample_method_ == 'random':
                        index_ = list(self.data_dict[k][phase_]['y'] == label_)
                        sample_list = []
                        for i_, is_true in enumerate(index_):
                            if is_true:
                                sample_list.append(self.data_dict[k][phase_]['x'][i_])
                        r = np.arange(len(sample_list))
                        np.random.shuffle(r)
                        r = list(r[:v])
                        re_list.extend([transform_action(sample_list[x]).unsqueeze(0) if self.load_to_ram else image_load_preprocess(sample_list[x], transform_action).unsqueeze(0) for x in
                                        r])
                    elif sample_method_ == 'herding':
                        label_sample = self._get_image_array(self.data_dict[k][phase_]['x'],
                                                             self.data_dict[k][phase_]['y'] == label_)
                        feature_sample, label_sample = self._get_feature(model_, label_sample, 'cpu', transform_action)
                        r = _herding_process(feature_sample, v, herding_method)
                        r = list(r[:v])
                        re_list.extend([label_sample[x].unsqueeze(0) for x in r])
                    else:
                        raise ValueError('invalid sample method {}'.format(sample_method_))
                    re_list_label.extend([label_.unsqueeze(0)] * v)
            else:
                r = np.arange(len(self.data_dict[k][phase_]['y']))
                np.random.shuffle(r)
                r = list(r[:v])
                re_list.extend([transform_action(self.data_dict[k][phase_]['x'][x]).unsqueeze(0) if self.load_to_ram else image_load_preprocess(self.data_dict[k][phase_]['x'][x], transform_action).unsqueeze(0) for x in r])
                re_list_label.extend([self.data_dict[k][phase_]['y'][x].clone().detach().unsqueeze(0) for x in r])
        return torch.cat(re_list, dim=0), torch.cat(re_list_label, dim=0)


    def sample_selection(self, num_dict_: dict, phase_: int, select_sample_per_class_: bool, sample_method_: str = 'random', model_=None, device_='cpu', herding_method='l2', random_seed=None):
        """
        select samples for svd
        herding here simply use top-k features closest to the prototype
        if herding is used, select sample for each class
        :param random_seed:
        :param herding_method: l2/cos
        :param device_:
        :param model_:
        :param select_sample_per_class_:
        :param phase_:
        :param sample_method_: ('random', 'herding')
        :param num_dict_:
        :return: sample tensor
        """
        re_list = []
        re_list_label = []
        if random_seed is None:
            random_seed = self.random_seed
        np.random.seed(random_seed)
        for k, v in num_dict_.items():
            if k not in self.data_dict.keys():
                continue
            transform_action = self.transform_dict['train'] if k == 'train' and self.training_set_transform else self.transform_dict['test']
            if select_sample_per_class_:
                for label_ in torch.unique(self.data_dict[k][phase_]['y']):
                    if sample_method_ == 'random':
                        index_ = list(self.data_dict[k][phase_]['y'] == label_)
                        sample_list = []
                        for i_, is_true in enumerate(index_):
                            if is_true:
                                sample_list.append(self.data_dict[k][phase_]['x'][i_])
                        r = np.arange(len(sample_list))
                        np.random.shuffle(r)
                        r = list(r[:v])
                        re_list.extend([transform_action(sample_list[x]).unsqueeze(0) if self.load_to_ram else image_load_preprocess(sample_list[x], transform_action).unsqueeze(0) for x in r])
                    elif sample_method_ == 'herding':
                        label_sample = self._get_image_array(self.data_dict[k][phase_]['x'], self.data_dict[k][phase_]['y'] == label_)
                        feature_sample, label_sample = self._get_feature(model_, label_sample, 'cpu', transform_action)
                        r = _herding_process(feature_sample, v, herding_method)
                        r = list(r[:v])
                        re_list.extend([label_sample[x].unsqueeze(0) for x in r])
                    else:
                        raise ValueError('invalid sample method {}'.format(sample_method_))
                    re_list_label.extend([label_.unsqueeze(0)] * v)
            else:
                r = np.arange(len(self.data_dict[k][phase_]['y']))
                np.random.shuffle(r)
                r = list(r[:v])
                re_list.extend([transform_action(self.data_dict[k][phase_]['x'][x]).unsqueeze(0) if self.load_to_ram else image_load_preprocess(self.data_dict[k][phase_]['x'][x], transform_action).unsqueeze(0) for x in r])
                re_list_label.extend([self.data_dict[k][phase_]['y'][x].clone().detach().unsqueeze(0) for x in r])
        return torch.cat(re_list, dim=0), torch.cat(re_list_label, dim=0)

    def get_current_sets_length(self):
        if self.mode[self.current_set] == 'up_to_now':
            self.gather_up_to_now_data()
            return len(self.up_to_now_data[self.current_set]['y'])
        elif self.mode[self.current_set] == 'current_phase':
            return len(self.data_dict[self.current_set][self.current_phase]['y'])
        else:
            self.gather_joint_data()
            return len(self.joint_data[self.current_set]['y'])

    def __getitem__(self, item):
        if self.mode[self.current_set] == 'up_to_now':
            if not self.load_to_ram:
                image = image_load_preprocess(self.up_to_now_data[self.current_set]['x'][item], self.transform_dict[self.current_set])
            else:
                image = self.transform_dict[self.current_set](self.up_to_now_data[self.current_set]['x'][item])
            label = self.up_to_now_data[self.current_set]['y'][item]
            return image, label
        elif self.mode[self.current_set] == 'joint':
            if not self.load_to_ram:
                image = image_load_preprocess(self.joint_data[self.current_set]['x'][item], self.transform_dict[self.current_set])
            else:
                image = self.transform_dict[self.current_set](self.joint_data[self.current_set]['x'][item])
            label = self.joint_data[self.current_set]['y'][item]
            return image, label
        else:
            if not self.load_to_ram:
                image = image_load_preprocess(self.data_dict[self.current_set][self.current_phase]['x'][item], self.transform_dict[self.current_set])
            else:
                image = self.transform_dict[self.current_set](self.data_dict[self.current_set][self.current_phase]['x'][item])
            label = self.data_dict[self.current_set][self.current_phase]['y'][item]
            return image, label

    def __len__(self):
        if self.mode[self.current_set] == 'up_to_now':
            return len(self.up_to_now_data[self.current_set]['y'])
        elif self.mode[self.current_set] == 'current_phase':
            return len(self.data_dict[self.current_set][self.current_phase]['y'])
        else:
            return len(self.joint_data[self.current_set]['y'])


class FundusDILDataset(BaseDataset):
    def __init__(self, dir_list_, disease_list_, resize_transform_, **kwargs):
        super(FundusDILDataset, self).__init__(**kwargs)
        self.dir_list = dir_list_
        self.disease_list = disease_list_
        self.task_num = len(dir_list_)
        self.total_nc = len(disease_list_)
        self.resize_transform = resize_transform_
        self.disease_to_label_dict, self.label_to_disease_dict = self._get_disease_to_label_mapping_dict()
        self.data_dict = self.data_dict_obtain_()

    def _get_disease_to_label_mapping_dict(self):
        disease_to_label_dict = {}
        label_to_disease_dict = {}
        for idx, disease in enumerate(self.disease_list):
            disease_to_label_dict[disease] = idx
            label_to_disease_dict[idx] = disease
        return disease_to_label_dict, label_to_disease_dict

    def data_dict_obtain_(self):
        data_dict = {'train': {}, 'test': {}, 'val': {}}
        use_validation_flag = 1
        for phase, set_dir in enumerate(self.dir_list):
            for set_name in data_dict.keys():
                if phase not in data_dict[set_name].keys():
                    data_dict[set_name][phase] = {'x': [], 'y': []}
                for label_name in self.disease_list:
                    if set_name in os.listdir(set_dir):
                        data_path = os.path.join(set_dir, set_name, label_name)
                        label_id = self.disease_to_label_dict[label_name]
                        for image_name in sorted(os.listdir(data_path)):
                            image_path = os.path.join(data_path, image_name)
                            data_dict[set_name][phase]['y'].append(torch.tensor([label_id], dtype=torch.long))
                            if self.load_to_ram:
                                if len(data_dict[set_name][phase]['x']) == 0:
                                    data_dict[set_name][phase]['x'] = np.expand_dims(image_load_preprocess(image_path, self.resize_transform), axis=0)
                                else:
                                    data_dict[set_name][phase]['x'] = np.concatenate([data_dict[set_name][phase]['x'], np.expand_dims(image_load_preprocess(image_path, self.resize_transform), axis=0)], axis=0)
                            else:
                                data_dict[set_name][phase]['x'].append(image_path)
                    elif set_name in ('train', 'test'):
                        raise AssertionError('no {} under {}'.format(set_name, set_dir))
                    else:
                        use_validation_flag = 0
                data_dict[set_name][phase]['y'] = torch.cat(data_dict[set_name][phase]['y'], dim=0)
        self.use_validation = True if use_validation_flag == 1 else False
        return data_dict
