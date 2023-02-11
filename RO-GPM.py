import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
from Baseline import BaselineDIL
from basic_functions import dict_append, check_and_make_dir, acc_matrix_to_dict, acc_matrix_print, dict_save, model_selection, dict_padding, _check_resized_dataset, occupy_memory, forgetting_measure
from sklearn.metrics import classification_report, cohen_kappa_score, balanced_accuracy_score
from skimage.io import imshow
import matplotlib.pyplot as plt
from Analysis import ModelAnalysisDIL

# import multiprocessing as mp
import time
from Datasets import TempDataset, InfiniteBatchSampler
from torch.utils.data import DataLoader


def eigen_decomposition(matrix_tensor_):
    eigen_val, u_ = np.linalg.eig(torch.mm(matrix_tensor_, matrix_tensor_.permute(1, 0)).numpy())
    eigen_val, u_ = torch.tensor(eigen_val, dtype=torch.float), torch.tensor(u_, dtype=torch.float)
    _, descending_index = torch.sort(eigen_val, dim=0, descending=True)
    eigen_val, u_ = eigen_val[descending_index].clone(), u_[:, descending_index].clone()
    return eigen_val, u_


def _projection(param_, u_, singular_, reference_param_, layer_idx_, mp_dict_):
    size_ = param_.size(0)
    k_size = u_.size(1)
    # diag version
    norm_grad = param_.view(size_, -1)
    norm_reference_grad = reference_param_.view(size_, -1).permute(1, 0)
    # [d0 d] [d k] [k d] [d d0]->[d0 d0]
    grad_projection = torch.diag(torch.mm(torch.mm(torch.mm(norm_grad, u_), torch.diag(singular_)), torch.mm(u_.permute(1, 0), norm_reference_grad)))

    grad_projection[grad_projection > 0] = 0
    grad_projection[grad_projection < 0] = 1
    grad_projection = grad_projection.unsqueeze(1).expand(-1, k_size)
    projection_grad = torch.mm(torch.mm(norm_grad, u_) * grad_projection, u_.permute(1, 0))
    mp_dict_[layer_idx_] = projection_grad.view(param_.size())


class ROGPM(BaselineDIL):
    def __init__(self, args_, model_, device_):
        super(ROGPM, self).__init__(args_, model_, device_)
        self.old_model = None
        self.selected_sample_num_dict = {'train': args_.sample_selection_num[0], 'val': args_.sample_selection_num[1], 'test': args_.sample_selection_num[2]}
        self.total_selected_sample_num_ = int(sum(args_.sample_selection_num))
        self.u_list = None
        self.singular_list = None
        self.threshold_list = self._get_threshold_list()
        self.temp_feature_in_list_ = []
        self.feature_in_list_ = []
        self.temp_buffer_dict = {'x': [], 'y': [], 'current_size': 0, 'buffer_size': 0, 'reservoir_sample_num': 0}
        self.handle_list = None
        self.total_capacity = None
        self.save_path = self.exp_path_obtain(-1, exp_name='ro-gpm')
        self.gpm_memory_list = []
        self.buffer_sample_list = []
        if self.args.projection_device == 'cpu':
            self.calculation_device = 'cpu'
            self.store_device = 'cpu'
        elif self.args.projection_device == 'gpu':
            self.calculation_device = self.device
            self.store_device = self.device
        else:
            raise NotImplementedError(self.args.projection_device)
        print('projection device = {}'.format(self.args.projection_device))


    def _get_reduced_dataset_order(self):
        order_list = []
        for set_path in self.args.dataset_path[: -1]:
            order_list.append(os.path.basename(set_path))
        return '-'.join(order_list)

    def rogpm_memory_list_save(self):
        dict_save({'gpm_memory_ratio': self.gpm_memory_list}, 'gpm_mem_used_ratio', self.save_path)

    def _get_threshold_list(self):
        threshold_list_length = len(self.args.threshold)
        total_phase_ = self.dataset.total_phase_obtain()
        if threshold_list_length >= total_phase_:
            return self.args.threshold[: total_phase_]
        else:
            pad_num = total_phase_ - threshold_list_length
            return self.args.threshold + [self.args.threshold[-1]] * pad_num

    def _remove_all_zero_column(self, tensor_):
        assert len(tensor_.size()) == 2, print('invalid tensor_ size {}'.format(tensor_.size()))
        return tensor_[:, torch.sum(tensor_, dim=0) != 0].detach()

    def reservoir_sampling(self, batch_data, batch_label):
        buffer_size = self.temp_buffer_dict['buffer_size']
        for idx_ in range(int(batch_label.size(0))):
            self.temp_buffer_dict['reservoir_sample_num'] += 1
            current_buffer_size = self.temp_buffer_dict['current_size']
            if current_buffer_size < buffer_size:
                self.temp_buffer_dict['x'].append(np.expand_dims(batch_data[idx_], axis=0))
                self.temp_buffer_dict['y'].append(batch_label[idx_].clone().detach().unsqueeze(0).cpu())
                self.temp_buffer_dict['current_size'] += 1
            else:
                p = torch.randint(0, self.temp_buffer_dict['reservoir_sample_num'], size=(1, ))
                if p < buffer_size:
                    self.temp_buffer_dict['x'][p] = np.expand_dims(batch_data[idx_], axis=0)
                    self.temp_buffer_dict['y'][p] = batch_label[idx_].clone().detach().unsqueeze(0).cpu()

    def forward_hook(self, module_, feature_in_, feature_out_):
        # unfold the input feature for conv layer
        for i_, feature_ in enumerate(feature_in_):
            if i_ >= 1:
                print('module {} has more than 1 input'.format(module_))
            if isinstance(module_, (nn.Conv2d,)):
                new_feature_ = nn.Unfold(module_.kernel_size, module_.dilation, module_.padding, module_.stride)(feature_).clone().detach()
                new_feature_ = new_feature_.permute(1, 0, 2).clone().detach().contiguous()
                new_feature_size_ = new_feature_.size()
                self.temp_feature_in_list_.append(self._remove_all_zero_column(new_feature_.view(new_feature_size_[0], -1)).cpu())
            else:
                self.temp_feature_in_list_.append(self._remove_all_zero_column(feature_.clone().detach().permute(1, 0)).cpu())
        return None

    def _feature_in_list_empty(self):
        self.feature_in_list_ = []

    def conv_linear_param_search(self, specified_model=None):
        """
        ignore last fc
        :return:
        """
        name_list_ = []
        param_list_ = []
        if specified_model is None:
            for name_, param_ in self.model.named_modules():
                if isinstance(param_, (nn.Conv2d, nn.Linear)):
                    name_list_.append(name_)
                    param_list_.append(param_)
        else:
            for name_, param_ in specified_model.named_modules():
                if isinstance(param_, (nn.Conv2d, nn.Linear)):
                    name_list_.append(name_)
                    param_list_.append(param_)
        return name_list_, param_list_

    def last_linear_layer_search(self):
        last_linear = None
        if isinstance(self.model, nn.Module):
            for name_, param_ in self.model.named_modules():
                if isinstance(param_, nn.Linear):
                    last_linear = param_
        return last_linear

    def model_forward_hook_register(self):
        handle_list_ = []
        _, param_list_ = self.conv_linear_param_search()
        for i_, param_ in enumerate(param_list_):
            handle_list_.append(param_.register_forward_hook(hook=self.forward_hook))
        return handle_list_

    def _get_feature_in_list_size(self):
        if not self.feature_in_list_:
            return 0
        total_size = 0
        for t_ in self.feature_in_list_:
            total_size += t_.size(0) * t_.size(1)
        return total_size * 4 / 1024 / 1024 / 1024

    def representation_matrix_obtain(self, selected_samples_):
        """
        features will be directly appended to global variable 'feature_in_list'
        :param selected_samples_:
        :return:
        """
        self._feature_in_list_empty()
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            self.handle_list = self.model_forward_hook_register()
            for idx in range(int(np.ceil(selected_samples_.size(0) / float(self.args.batch_size)))):
                start_ = idx * self.args.batch_size
                end_ = start_ + self.args.batch_size
                _ = self.model(selected_samples_[start_: end_].to(self.device))
                if not self.feature_in_list_:
                    self.feature_in_list_ = self.temp_feature_in_list_
                else:
                    for idx_ in range(len(self.feature_in_list_)):
                        self.feature_in_list_[idx_] = torch.cat([self.feature_in_list_[idx_], self.temp_feature_in_list_[idx_]], dim=1)
                self.temp_feature_in_list_ = []
            print('current feature mem = {}GB'.format(self._get_feature_in_list_size()))
        self._handle_empty()

    def _handle_empty(self):
        for handle_ in self.handle_list:
            handle_.remove()

    def _singular_value_ratio_plot(self, singular_val_ratio_, layer_num_):
        save_p = check_and_make_dir(os.path.join(self.save_path, 'singular_value_plot'))
        value_save_p = check_and_make_dir(os.path.join(self.save_path, 'singular_value'))
        plt.figure()
        plt.plot(range(1, int(singular_val_ratio_.squeeze().size(0)) + 1), np.array(singular_val_ratio_.squeeze().cpu().numpy()), marker='x', linewidth=1.5, color='b')
        plt.savefig(os.path.join(save_p, 'layer_{}.png'.format(layer_num_)))
        plt.close()
        torch.save(singular_val_ratio_, os.path.join(value_save_p, 'layer_{}.pth'.format(layer_num_)))

    def projection_update(self, phase_):
        threshold_ = self.threshold_list[phase_]
        total_capacity = 0
        current_capacity = 0
        if self.u_list is None:
            self.u_list = []
            self.singular_list = []
        with torch.no_grad():
            print('threshold = {}'.format(threshold_))
            if phase_ == 0:
                for idx, feature_ in enumerate(self.feature_in_list_):
                    eigen_val, u_ = eigen_decomposition(feature_.cpu())
                    # F-norm
                    singular_val_ratio = eigen_val / eigen_val.sum()
                    self.u_list.append(u_[:, : torch.sum(torch.cumsum(singular_val_ratio, dim=0) < threshold_)])
                    self.singular_list.append(singular_val_ratio[:torch.sum(torch.cumsum(singular_val_ratio, dim=0) < threshold_)])
                    total_capacity += u_.size(0) ** 2
            else:
                for idx, (u_old_, feature_) in enumerate(zip(self.u_list, self.feature_in_list_)):
                    feature_ = feature_.cpu()
                    eigen_val_, u_ = eigen_decomposition(feature_)
                    u_old_ = u_old_.cpu()
                    eigen_val_hat_, u_hat_ = eigen_decomposition((feature_ - torch.mm(torch.mm(u_old_, u_old_.clone().permute(1, 0)), feature_)).cpu())
                    singular_val_hat_ratio = eigen_val_hat_ / eigen_val_.sum()
                    singular_accumulated_val_ratio = 1 - (eigen_val_hat_.sum() / eigen_val_.sum())
                    r = 0
                    for i_ in range(singular_val_hat_ratio.shape[0]):
                        if singular_accumulated_val_ratio < threshold_:
                            singular_accumulated_val_ratio += singular_val_hat_ratio[i_]
                            r += 1
                        else:
                            break
                    self.u_list[idx] = torch.cat([u_old_, u_hat_[:, :r]], dim=1)[:, : u_old_.size(0)]
                    self.singular_list[idx] = torch.cat([self.singular_list[idx].cpu(), singular_val_hat_ratio[:r]], dim=0)[: u_old_.size(0)]
        if self.total_capacity is None:
            self.total_capacity = total_capacity * 4 / 1024 / 1024
        print('Representation Matrix (u_column/node_width)')
        print('=' * 50)
        for i_, u_ in enumerate(self.u_list):
            print('Layer {} : {}/{}'.format(i_ + 1, u_.size(1), u_.size(0)))
            current_capacity += u_.size(1) * u_.size(0)
        print('capacity use: {:.6f}MB/{:.6f}MB'.format(current_capacity * 4 / 1024 / 1024, self.total_capacity))
        print('=' * 50)
        self._feature_in_list_empty()

    def gradient_projection(self, reference_model):
        name_list_, param_list_ = self.conv_linear_param_search()
        reference_name_list_, reference_param_list_ = self.conv_linear_param_search(reference_model)
        assert len(param_list_) == len(self.u_list), print('inconsistent length of param_list and u_list')
        assert len(param_list_) == len(reference_param_list_), print('inconsistent length of param_list and reference_param_list')
        with torch.no_grad():
            for idx, (param_, reference_param_, u_, singular_) in enumerate(zip(param_list_, reference_param_list_, self.u_list, self.singular_list)):
                if hasattr(param_, 'weight'):
                    if param_.weight.grad is None:
                        continue
                    else:
                        u_ = u_.to(self.calculation_device)
                        singular_ = singular_.to(self.calculation_device)
                        if hasattr(param_, 'weight'):
                            if param_.weight.grad is None:
                                continue
                            else:
                                param_t, reference_param_t = param_.weight.grad.data.clone().detach().to(
                                    self.calculation_device), reference_param_.weight.grad.data.clone().detach().to(
                                    self.calculation_device)
                                mp_dict = {}
                                _projection(param_t, u_, singular_, reference_param_t, 0, mp_dict)
                                param_.weight.grad.data -= mp_dict[0].to(self.device)
                        if hasattr(param_, 'bias'):
                            if param_.bias is not None:
                                if param_.bias.grad is not None:
                                    param_.bias.grad.data.fill_(0)
                        self.u_list[idx] = self.u_list[idx].cpu()
                        self.singular_list[idx] = self.singular_list[idx].cpu()
            # process Norm Layer
            if self.args.fix_norm_affine:
                for _, param_ in self.model.named_modules():
                    if isinstance(param_, nn.BatchNorm2d):
                        if hasattr(param_, 'weight'):
                            if param_.weight is not None:
                                if param_.weight.grad is not None:
                                    param_.weight.grad.data.fill_(0)
                        if hasattr(param_, 'bias'):
                            if param_.bias is not None:
                                if param_.bias.grad is not None:
                                    param_.bias.grad.data.fill_(0)
            self.model.to(self.device)

    def last_conv_layer_search(self):
        last_conv = None
        if isinstance(self.model, nn.Module):
            for name_, param_ in self.model.backbone.named_modules():
                if isinstance(param_, nn.Conv2d):
                    last_conv = param_
        return last_conv

    def after_train(self, current_phase):
        self.temp_buffer_dict['buffer_size'] += self.args.buffer_size
        if current_phase != self.total_phase - 1:
            if self.args.buffer_type == 'mean':
                sample_num = int(self.args.buffer_size / len(self.args.disease_list))
                selection_per_class = True
            elif self.args.buffer_type == 'per_phase':
                sample_num = self.args.buffer_size
                selection_per_class = False
            else:
                raise NotImplementedError
            selected_raw_sample, selected_raw_label = self.dataset.origin_sample_selection({'train': sample_num, 'val': 0, 'test': 0}, current_phase, selection_per_class, 'random')
            self.reservoir_sampling(selected_raw_sample, selected_raw_label)
            print('current buffer size = {}, max = {}'.format(self.temp_buffer_dict['current_size'], self.temp_buffer_dict['buffer_size']))
        if current_phase != self.total_phase - 1:
            selected_sample, selected_label = self.dataset.sample_selection(self.selected_sample_num_dict, current_phase, self.args.select_sample_per_class, self.args.sample_selection_method, self.model, self.device, 'l2', self.args.gpm_random_seed)
            print('selected samples num = {}, selected classes = {}'.format(selected_sample.size(0), torch.unique(selected_label)))
            self.representation_matrix_obtain(selected_sample)
            self.projection_update(current_phase)
        model_path_ = os.path.join(self.save_path, 'model_{}.pkl'.format(str(current_phase)))
        if self.save_model is not None:
            self.model = self.save_model
        self.model.to('cpu')
        torch.save(self.model, model_path_)

    def _deepcopy_current_model(self):
        model_ = copy.deepcopy(self.model)
        model_.eval()
        model_.to(self.calculation_device)
        return model_

    def _buffer_sample_list_save(self, buffer_sample_):
        if buffer_sample_.size(0) < self.args.batch_size:
            return
        self.buffer_sample_list.append(buffer_sample_.unsqueeze(0))
        np.savetxt(os.path.join(self.save_path, 'buffer.txt'), torch.cat(self.buffer_sample_list, dim=0).numpy(), fmt='%d')

    def current_set_gradient_computation(self, current_batch_image, current_batch_label):
        temp_model = self._deepcopy_current_model()
        temp_model.zero_grad()
        mix_sample, mix_label = current_batch_image.to(self.calculation_device), current_batch_label.to(self.calculation_device)
        mix_out = temp_model(mix_sample)
        mix_loss = self.get_loss()(mix_out, mix_label)
        mix_loss.backward()

        return temp_model

    def _compute_loss(self, image, label, phase_, buffer_dataloader=None):
        batch_size = label.size(0)
        if phase_ > 0:
            transform_replay_sample, transform_replay_label = buffer_dataloader.__iter__().__next__()
            mix_sample, mix_label = torch.cat([image, transform_replay_sample.to(self.device)], dim=0), torch.cat(
                [label, transform_replay_label.to(self.device)], dim=0)
        else:
            mix_sample, mix_label = image, label
        pred = self.model(mix_sample)
        criterion = self.get_loss()
        loss_cls = criterion(pred / self.train_temperature, mix_label)
        correct_iter = (torch.argmax(pred[:batch_size, :].clone().detach(), dim=1) == label.clone().detach()).sum().cpu().numpy()
        return loss_cls, correct_iter

    def load_check_point(self, optimizer_, load_all=False):
        check_point = torch.load(os.path.join(self.save_path, 'check_point.pth'))
        if not load_all:
            return check_point['current_phase']
        model_ = check_point['model'].to(self.device)
        optimizer_.load_state_dict(check_point['optimizer'])
        return model_, optimizer_, check_point['next_epoch']

    def get_corresponding_dataloader(self):
        replay_sample, replay_label = np.concatenate(self.temp_buffer_dict['x']), torch.cat(self.temp_buffer_dict['y'], dim=0)
        load_to_ram = False
        return DataLoader(
            TempDataset(transforms.Compose(self.dataset.transform_dict['train'].transforms), replay_sample,
                        replay_label, load_to_ram=load_to_ram), batch_sampler=InfiniteBatchSampler(self.temp_buffer_dict['current_size'],
                                                                          self.dataset.get_current_sets_length(),
                                                                          batch_size=self.args.batch_size), num_workers=32)

    def train(self, phase_):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        best_val_correct = 0
        patience = self.args.patience
        start_epoch = 0
        if self.args.resume and self.load_check_point(None, False) <= phase_:
            self.model, optimizer, start_epoch = self.load_check_point(optimizer, True)
        for epoch in range(start_epoch, self.args.epochs):
            total = 0
            correct_all = 0
            loss_cls_all = 0
            if phase_ > 0:
                buffer_dataloader = self.get_corresponding_dataloader()
            else:
                buffer_dataloader = None
            self.model.train()
            if self.args.fix_norm_affine and phase_ > 0:
                self.model.freeze_bn()
            for idx, (image, label) in enumerate(self.reset_dataset_and_get_dataloader(phase_, 'train', self.training_mode, True)):
                image, label = image.to(self.device), label.to(self.device)
                total += label.size(0)
                optimizer.zero_grad()
                loss_iter, correct_iter = self._compute_loss(image, label, phase_, buffer_dataloader)
                loss_iter.backward()
                if phase_ > 0:
                    self.gradient_projection(self.current_set_gradient_computation(image, label))
                optimizer.step()
                loss_cls_all += loss_iter.clone().detach().cpu().numpy() * label.size(0)
                correct_all += correct_iter
            print('phase[{}/{}], epoch[{}/{}]: train acc = {:.4f}, loss = {:.4f}\t'.format(phase_ + 1, self.total_phase, epoch + 1, self.args.epochs, correct_all / total, loss_cls_all / total), end=' ')
            if self.dataset.use_validation:
                val_loss, val_correct, val_total, val_prediction, val_label = self.model_eval(self.model, phase_, 'val',
                                                                                              self.training_mode)
                val_metric = self._metrics_results_obtain(val_label, val_prediction)
                print('val {} = {:.4f}, val loss = {:.4f}\t'.format(self.args.best_model_metric, val_metric,
                                                                    val_loss / val_total), end=' ')
                if val_metric > best_val_correct:
                    print('*\t', end=' ')
                validation_return = self._validation(val_metric, best_val_correct, patience, optimizer)
                if validation_return is None:
                    print('\nearly stop at epoch {}'.format(epoch + 1))
                    break
                else:
                    best_val_correct, patience = validation_return
            elif scheduler is not None:
                scheduler.step()
                print('lr = {}\t'.format(optimizer.param_groups[0]['lr']), end=' ')
            if epoch % self.args.test_frequency == 0:
                test_loss, test_correct, test_total, test_prediction, test_label = self.model_eval(self.model, phase_,
                                                                                                   'test',
                                                                                                   self.training_mode)
                test_metric = self._metrics_results_obtain(test_label, test_prediction)
                print('test {} = {:.4f}, loss = {:.4f}\t'.format(self.args.best_model_metric, test_metric,
                                                                 test_loss / test_total), end=' ')
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RO-GPM')
    # base
    parser.add_argument('--save_name', type=str, default=None)
    # parser.add_argument('--mem_require', type=float, default=0)
    parser.add_argument('--baseline_mode', type=str, default='FT', help='FT(fine tune)')
    parser.add_argument('--evaluation', action='store_true', default=False)
    parser.add_argument('--cuda_id', type=str, default='2')
    parser.add_argument('--use_ft_pretrained_network', action='store_true', default=False,
                        help='whether to use network trained on phase1 in FT experiment, this can ensure base network is the same')
    parser.add_argument('--use_sub_exp_ft', action='store_true', default=False, help='if True, only train unseen domain, use sub-experiments last model')
    parser.add_argument('--resume', action='store_true', default=False)
    # GPM settings
    parser.add_argument('--threshold', type=float, nargs='*', default=[0.99], help='GPM threshold for each phase, need to repeat when length not accordance with task_num + 1')
    parser.add_argument('--sample_selection_num', nargs=3, type=int, default=[80, 0, 0], help='follow the order of train val test')
    parser.add_argument('--select_sample_per_class', action='store_true', default=True, help='if True, then for each sample_selection_num, select sample_selection_num samples per class')
    parser.add_argument('--sample_selection_method', type=str, default='random', help='random')
    parser.add_argument('--fix_norm_affine', action='store_true', default=True)
    parser.add_argument('--buffer_size', type=int, default=400, help='memory buffer size')
    parser.add_argument('--buffer_type', type=str, default='mean', help='all=totally num/per_phase=random selection for each phase/mean=same for each class')
    parser.add_argument('--projection_device', type=str, default='gpu', help='cpu/gpu')
    # network
    parser.add_argument('--network_architecture', type=str, default='resnet18')
    parser.add_argument('--use_pretrained_backbone', action='store_true', default=True)
    parser.add_argument('--best_model_metric', type=str, default='macro_f1')
    # dataset
    parser.add_argument('--dataset_name', type=str, default='fundus', help='fundus')
    parser.add_argument('--disease_list', type=str, nargs='*', default=['normal', 'glaucoma', 'amd', 'dr', 'myopia'])
    parser.add_argument('--dataset_path', type=str, nargs='*', default=['./ODIR_ALL', './ISEE_ALL'], help='')

    parser.add_argument('--resize_shape', type=int, default=512)
    parser.add_argument('--load_to_ram', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=4)
    parser.add_argument('--gpm_random_seed', type=int, default=1)
    # learning rate
    parser.add_argument('--lr', type=float, default=0.001)
    # train settings
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam')
    parser.add_argument('--scheduler', type=str, default='steplr')
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--steplr_gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default='5e-4')
    parser.add_argument('--train_temperature', type=float, default=1)
    parser.add_argument('--loss', type=str, default='ce')
    # validation settings
    parser.add_argument('--lr_min', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr_decay_factor', type=float, default=10)
    # test settings
    parser.add_argument('--test_frequency', type=int, default=1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.dataset_name == 'fundus':
        args.dataset_path = _check_resized_dataset(args.dataset_path, args.resize_shape)
    print(args)

    network = model_selection(args.dataset_name, len(args.disease_list), args.network_architecture, use_pretrained_backbone_=args.use_pretrained_backbone)
    model = ROGPM(args, network, device)

    total_phase = model.total_phase
    check_and_make_dir(model.save_path)
    dict_save(dict_padding(vars(args)), 'setting', model.save_path)

    # train
    if not args.evaluation:
        for i in range(total_phase):
            model.before_train(i)
            if i != total_phase - 1 and args.use_sub_exp_ft and os.path.exists(model.exp_path_obtain(None, exp_name='ro-gpm')) and total_phase > 2:
                print('use pretrain sub exp')
                path = model.exp_path_obtain(None, exp_name='ro-gpm')
                model.model = model.get_corresponding_model(i, path)
                print('phase {} load model from {}'.format(i, path))
            else:
                if i == 0 and args.use_ft_pretrained_network and model.baseline_path_obtain() != model.save_path:
                    path = model.baseline_path_obtain()
                    model.model = model.get_corresponding_model(i, path)
                else:
                    if not args.resume:
                        model.train(i)
                    else:
                        check_point_phase = model.load_check_point(None, False)
                        if check_point_phase <= i:
                            model.train(i)
            model.after_train(i)
    model.rogpm_memory_list_save()

    model_evaluation = ModelAnalysisDIL(args, model)
    model_evaluation.specific_test('test')

