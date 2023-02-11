import os
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
from basic_functions import dict_append, check_and_make_dir, acc_matrix_to_dict, acc_matrix_print, dict_save, dict_padding, model_selection, _check_resized_dataset, occupy_memory, forgetting_measure, fwt_measure
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models.resnet import BasicBlock, Bottleneck
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, cohen_kappa_score, balanced_accuracy_score
# from cam_results_DIL import single_exp_cam_cal_save, singe_exp_cam_iou_cal
# from kappa import quadratic_weighted_kappa


class ModelAnalysisDIL:
    def __init__(self, args_, model_):
        """
        feature_map_dict = {'class': {'phase': [feature_map_layer1(ncwh), ...]}}
        base_phase_feature_map_dict = {'phase': {'class': [feature_map_layer1(ncwh), ...]}}
        base_phase_feature_map_save_dict = {'phase': {'feature': [feature_map_layer1(ncwh), ...], 'label': torch.tensor(n)}} -> {'phase': {'class': [feature_map_layer1(ncwh), ...], 'label': torch.tensor(n)}}
        :param args_:
        :param model_:
        """
        self.args = args_
        self.model = model_
        self.model.device = model_.device
        self.feature_map_dict = {}
        self.base_phase_feature_map_dict = {}
        self.base_phase_feature_map_save_dict = {}
        self.feature_temp_list = []
        self.handle_list = []
        self.prototype_dict = {'train': {}, 'test': {}}
        self.split_num = self.args.rotation_num if 'self_supervise' in vars(self.args).keys() and self.args.self_supervise else 1
        self.cmap = ('blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan')

    def _fc_norm_calculation(self, net_):
        fc_ = net_.fc
        norm_weight = torch.linalg.norm(fc_.weight.data, dim=1)
        # if fc_.bias is not None:
        norm_bias = fc_.bias
        return norm_weight, norm_bias

    def test_for_each_phase(self, set_name_='test'):
        test_mode = 'current_phase' if self.args.baseline_mode != 'JTA' else 'joint'
        total_phase = self.model.total_phase if self.args.baseline_mode != 'JTA' else 1
        macro_f1_matrix = np.zeros((total_phase, total_phase))
        each_phase_metric_dict = {}
        for i in range(total_phase):
            network = self.model.get_corresponding_model(i)
            for j in range(total_phase):
                _loss, _correct, _total, _prediction, _label = self.model.model_eval(network, j, set_name_, test_mode)
                each_phase_metric_dict['{}_{}'.format(i, j)] = classification_report(y_true=_label, y_pred=_prediction,
                                                                                     target_names=self.args.disease_list if self.args.dataset_name == 'fundus' else list(range(10)),
                                                                                     zero_division=0, output_dict=True)
                # print(each_phase_metric_dict['{}_{}'.format(i, j)])
                macro_f1_matrix[i, j] = each_phase_metric_dict['{}_{}'.format(i, j)]['macro avg']['f1-score']
        torch.save(each_phase_metric_dict, os.path.join(self.model.save_path, 'classification_report_dict_all.pth'))
        macro_f1_forgetting = forgetting_measure(macro_f1_matrix)
        print('forgetting: {}'.format(macro_f1_forgetting))
        dict_save({'macro_f1_forgetting': [macro_f1_forgetting]}, 'avg_forgetting', self.model.save_path)
        dict_save({'macro_f1_fwt': [fwt_measure(macro_f1_matrix)]}, 'fwt', self.model.save_path)

        dict_save(acc_matrix_to_dict(macro_f1_matrix), 'test_macro_f1', self.model.save_path)

        task_avg_metric_dict = {}
        dict_append(task_avg_metric_dict, 'task_avg_macro_f1', np.mean(macro_f1_matrix[-1]))
        dict_save(task_avg_metric_dict, 'task_avg_metric', self.model.save_path)

    def specific_test(self, set_name_='test'):
        self.test_for_each_phase(set_name_)


