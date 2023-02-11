import numpy as np
import pandas as pd
import copy
import random
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from Datasets import FundusDILDataset
import argparse
import time
import os
from basic_functions import dict_append, check_and_make_dir, acc_matrix_to_dict, acc_matrix_print, dict_save, dict_padding, model_selection, _check_resized_dataset, occupy_memory, forgetting_measure
from sklearn.metrics import f1_score, accuracy_score
from Analysis import ModelAnalysisDIL


class BaselineDIL:
    def __init__(self, args_, model_, device_):
        self.device = device_
        self.train_temperature = args_.train_temperature
        self.model = model_
        self.args = args_
        self.training_mode = self._get_training_mode()
        self.dataset = self._get_dataset()
        self._set_random_seed()
        self.total_phase = self.total_phase = self.dataset.total_phase_obtain() if self.training_mode != 'joint' else 1
        self.dataset_order = self._get_sub_dataset_order(-1)
        self.save_model = None
        self.save_path = self.exp_path_obtain(sub_num=-1, exp_name='baseline')

    def _set_random_seed(self):
        seed_ = self.dataset.random_seed
        random.seed(seed_)
        np.random.seed(seed_)
        torch.manual_seed(seed_)
        torch.cuda.manual_seed(seed_)
        torch.cuda.manual_seed_all(seed_)

    def exp_path_obtain(self, sub_num=-1, baseline_mode=None, exp_name='baseline'):
        if baseline_mode is None:
            baseline_mode = self.args.baseline_mode.upper()
        exp_name = exp_name.lower()
        if exp_name == 'baseline':
            return os.path.join(os.getcwd(), 'experiment_baseline_DIL', 'D_{}_C{}_{}_{}_{}_{}'.format(
                   self._get_sub_dataset_order(sub_num) if self.training_mode != 'joint' else self.dataset.task_num, self.dataset.total_nc, baseline_mode, self.args.resize_shape, self.args.network_architecture,
                   self.args.best_model_metric))
        if exp_name == 'ro-gpm':
            return os.path.join(os.getcwd(), 'experiment_RO-GPM_DIL', 'D_{}_C{}_{}_{}_{}_{}_{}_{}'.format(
            self._get_sub_dataset_order(sub_num) if self.training_mode != 'joint' else self.dataset.task_num, self.dataset.total_nc,
            self.args.baseline_mode.upper(), self.args.resize_shape, self.args.network_architecture, 'GPM{}_{}{}{}'.format(self.args.threshold[0], self.args.sample_selection_method, self.args.sample_selection_num[0], 'c' if self.args.select_sample_per_class else 't'),
            'R{}_{}_{}'.format(self.args.buffer_size, self.args.buffer_type, 'matrix'), self.args.best_model_metric))

    def baseline_path_obtain(self):
        return self.exp_path_obtain(-1, 'FT', 'baseline')

    def _get_sub_dataset_order(self, sub_num=None):
        """
        None return [:-1], -1 return all, >0 return [: sub_num]
        :param sub_num:
        :return:
        """
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

    def _get_dataset_order(self):
        order_list = []
        for set_path in self.args.dataset_path:
            order_list.append(os.path.basename(set_path))
        return '-'.join(order_list)

    def _get_training_mode(self):
        bm = self.args.baseline_mode.upper()
        assert bm in ('FT', 'JT', 'JTA'), print('invalid baseline mode {}'.format(bm))
        if bm == 'FT':
            return 'current_phase'
        if bm == 'JT':
            return 'up_to_now'
        if bm == 'JTA':
            return 'joint'

    def _get_dataset(self):
        if self.args.dataset_name.lower() == 'fundus':
            train_transform = transforms.Compose([transforms.ToTensor(), transforms.RandomResizedCrop(self.args.resize_shape, scale=(0.8, 1), ratio=(1, 1)), transforms.ColorJitter(brightness=0.2), transforms.RandomHorizontalFlip(p=0.5)])
            test_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((self.args.resize_shape, self.args.resize_shape))])
            return FundusDILDataset(self.args.dataset_path, self.args.disease_list, transforms.Compose([transforms.Resize((self.args.resize_shape, self.args.resize_shape))]), train_transform_=train_transform, test_transform_=test_transform, load_to_ram_=self.args.load_to_ram, random_seed=self.args.random_seed)

    def before_train(self, current_phase):
        self.save_model = None
        self.model.eval()
        self.model.to(self.device)

    def get_corresponding_model(self, phase_, path_=None):
        # model_ = model_selection(args.dataset_name, self.dataset.get_current_phase_total_class(phase_))
        model_ = torch.load(os.path.join(self.save_path if path_ is None else path_, 'model_{}.pkl'.format(str(phase_))))
        # model_.load_state_dict(torch.load(os.path.join(self.save_path, 'model_{}.pth'.format(str(self.dataset.get_current_phase_total_class(phase_))))))
        model_.to(self.device)
        return model_

    def after_train(self, current_phase):
        if self.save_model is not None:
            self.model = self.save_model
        self.model.to('cpu')
        torch.save(self.model, os.path.join(self.save_path, 'model_{}.pkl'.format(str(current_phase))))

    def get_optimizer(self):
        if self.args.optimizer.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=0)
        if self.args.optimizer.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

    def get_scheduler(self, optimizer):
        if self.dataset.use_validation:
            return None
        if self.args.scheduler.lower() == 'steplr':
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.steplr_gamma)

    def get_loss(self):
        if self.args.loss.lower() == 'ce':
            return nn.CrossEntropyLoss()

    def reset_dataset_and_get_dataloader(self, phase_, set_name_, mode_, shuffle_, **kwargs):
        self.dataset.reset_all_and_prepare_data(set_name_, mode_, phase_)
        return DataLoader(self.dataset, batch_size=self.args.batch_size, shuffle=shuffle_, num_workers=4, **kwargs)

    def _compute_loss(self, image, label, phase_):
        pred = self.model(image)
        criterion = self.get_loss()
        loss_cls = criterion(pred / self.train_temperature, label)
        loss_cls.backward()
        correct_iter = (torch.argmax(pred.clone().detach(), dim=1) == label.clone().detach()).sum().cpu().numpy()
        return loss_cls, correct_iter

    def model_eval(self, model_, phase_, set_name_, mode_):
        model_.eval()
        criterion = self.get_loss()
        ground_truth_tensor = None
        prediction_tensor = None
        with torch.no_grad():
            total = 0
            correct = 0
            loss = 0
            for idx, (image, label) in enumerate(self.reset_dataset_and_get_dataloader(phase_, set_name_, mode_, False)):
                image, label = image.to(self.device), label.to(self.device)
                pred = model_(image)
                loss += criterion(pred, label).clone().detach().cpu().numpy() * label.size(0)
                correct += (torch.argmax(pred.clone().detach(), dim=1) == label.clone().detach()).sum().cpu().numpy()
                total += label.size(0)
                if prediction_tensor is None:
                    prediction_tensor = torch.argmax(pred.clone().detach(), dim=1).cpu()
                    ground_truth_tensor = label.clone().detach().cpu()
                else:
                    prediction_tensor = torch.cat([prediction_tensor, torch.argmax(pred.clone().detach(), dim=1).cpu()], dim=0)
                    ground_truth_tensor = torch.cat([ground_truth_tensor, label.clone().detach().cpu()])
        return loss, correct, total, prediction_tensor.numpy(), ground_truth_tensor.numpy()

    def _validation(self, val_metric_, best_val_metric_, patience, optimizer):
        if val_metric_ > best_val_metric_:
            best_val_metric_ = val_metric_
            patience = self.args.patience
            self.save_model = copy.deepcopy(self.model)
        else:
            patience -= 1
            if patience <= 0:
                patience = self.args.patience
                if optimizer.param_groups[0]['lr'] / self.args.lr_decay_factor < self.args.lr_min:
                    return None
                else:
                    optimizer.param_groups[0]['lr'] /= self.args.lr_decay_factor
                    print('lr = {}'.format(optimizer.param_groups[0]['lr']), end=' ')
        return best_val_metric_, patience

    def _metrics_results_obtain(self, ground_truth, pred):
        if self.args.best_model_metric == 'w_f1':
            return f1_score(ground_truth, pred, average='weighted')
        if self.args.best_model_metric == 'macro_f1':
            return f1_score(ground_truth, pred, average='macro')
        if self.args.best_model_metric == 'acc':
            return accuracy_score(ground_truth, pred)

    def train(self, phase_):
        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)
        best_val_correct = 0
        patience = self.args.patience
        for epoch in range(self.args.epochs):
            total = 0
            correct_all = 0
            loss_cls_all = 0
            self.model.train() if phase_ == 0 or not self.args.fix_norm_affine else self.model.freeze_bn()
            for idx, (image, label) in enumerate(self.reset_dataset_and_get_dataloader(phase_, 'train', self.training_mode, True)):
                image, label = image.to(self.device), label.to(self.device)
                total += label.size(0)
                optimizer.zero_grad()
                loss_iter, correct_iter = self._compute_loss(image, label, phase_)
                loss_cls_all += loss_iter.clone().detach().cpu().numpy() * label.size(0)
                correct_all += correct_iter
                optimizer.step()
            print('phase[{}/{}], epoch[{}/{}]: train acc = {:.4f}, loss = {:.4f}\t'.format(phase_ + 1, self.total_phase, epoch + 1, self.args.epochs, correct_all / total, loss_cls_all / total), end=' ')
            if self.dataset.use_validation:
                val_loss, val_correct, val_total, val_prediction, val_label = self.model_eval(self.model, phase_, 'val', self.training_mode)
                val_metric = self._metrics_results_obtain(val_label, val_prediction)
                print('val {} = {:.4f}, val loss = {:.4f}\t'.format(self.args.best_model_metric, val_metric, val_loss / val_total), end=' ')
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
                test_loss, test_correct, test_total, test_prediction, test_label = self.model_eval(self.model, phase_, 'test', self.training_mode)
                test_metric = self._metrics_results_obtain(test_label, test_prediction)
                print('test {} = {:.4f}, loss = {:.4f}\t'.format(self.args.best_model_metric, test_metric, test_loss / test_total), end=' ')
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Baseline')
    # base
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--baseline_mode', type=str, default='FT', help='FT(fine tune)/JT(joint train)/JTA(joint train in a single phase)')
    parser.add_argument('--evaluation', action='store_true', default=False)
    parser.add_argument('--ft_load_base', action='store_true', default=False)
    parser.add_argument('--cuda_id', type=str, default='1')
    parser.add_argument('--use_sub_exp_ft', action='store_true', default=False)
    parser.add_argument('--use_ft_pretrained_network', action='store_true', default=True, help='whether to use network trained on phase1 in FT experiment, this can ensure base network is the same')
    # network
    parser.add_argument('--network_architecture', type=str, default='resnet18', help='resnet18 for Fundus')
    parser.add_argument('--use_pretrained_backbone', action='store_true', default=True, help='keep True for fast training')
    parser.add_argument('--best_model_metric', type=str, default='w_f1', help='w_f1/f1/acc/b_acc')
    parser.add_argument('--fix_norm_affine', action='store_true', default=False)
    parser.add_argument('--use_layer_norm', action='store_true', default=False)
    # dataset
    parser.add_argument('--dataset_name', type=str, default='fundus', help='fundus')
    parser.add_argument('--disease_list', type=str, nargs='*', default=['normal', 'glaucoma', 'amd', 'dr', 'myopia'])
    parser.add_argument('--dataset_path', type=str, nargs='*')
    parser.add_argument('--resize_shape', type=int, default=512)
    parser.add_argument('--load_to_ram', action='store_true', default=False)
    parser.add_argument('--random_seed', type=int, default=4)
    # learning rate
    parser.add_argument('--lr', type=float, default=0.001)
    # train settings
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--optimizer', type=str, default='sgd', help='sgd/adam')
    parser.add_argument('--scheduler', type=str, default='steplr')
    parser.add_argument('--step', type=int, default=60)
    parser.add_argument('--steplr_gamma', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default='5e-4')
    parser.add_argument('--train_temperature', type=float, default=1)
    parser.add_argument('--loss', type=str, default='ce')
    # validation settings
    parser.add_argument('--lr_min', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr_decay_factor', type=float, default=10)
    # test settings
    parser.add_argument('--test_frequency', type=int, default=5)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.dataset_path = _check_resized_dataset(args.dataset_path, args.resize_shape)
    network = model_selection(args.dataset_name, len(args.disease_list), args.network_architecture, use_pretrained_backbone_=args.use_pretrained_backbone)
    print(args)
    model = BaselineDIL(args, network, device)
    total_phase = model.total_phase
    check_and_make_dir(model.save_path)
    dict_save(dict_padding(vars(args)), 'setting', model.save_path)
    # train
    if not args.evaluation:
        for i in range(total_phase):
            model.before_train(i)
            if i != total_phase - 1 and args.use_sub_exp_ft and os.path.exists(model.exp_path_obtain(None, exp_name='baseline')) and total_phase > 2:
                print('use pretrain sub exp')
                path = model.exp_path_obtain(None, exp_name='baseline')
                model.model = model.get_corresponding_model(i, path)
                print('phase {} load model from {}'.format(i, path))
            else:
                if i == 0 and args.use_ft_pretrained_network and (args.ft_load_base or args.baseline_mode != 'FT') and os.path.exists(model.exp_path_obtain(None, exp_name='baseline')):
                    path = model.baseline_path_obtain()
                    print('Using baseline FT pretrained model')
                    model.model = model.get_corresponding_model(0, path)
                else:
                    model.train(i)
            model.after_train(i)
    model_evaluation = ModelAnalysisDIL(args, model)
    model_evaluation.specific_test('test')





