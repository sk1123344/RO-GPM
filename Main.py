import subprocess
import os
import time
import pynvml


def run_(program_, evaluation=False):
    if evaluation:
        print('evaluation')
        program_ += ' --evaluation'
    while True:
        try:

            re_code = subprocess.call(program_, shell=True)
            if re_code != 0:
                raise Exception
        except Exception as e:
            continue
        else:
            break


dataset_path_list = [('/data/shukuang/data/fundus/DIL712_preprocessed/ODIR_ALL /data/shukuang/data/fundus/DIL712_preprocessed/RIADD_REFUGE', False)]
network_architecture = 'resnet18'
epochs = 90
# epochs = 1
cuda_id = '2'
batch_size = 32


sample_method_list = ['random']
fix_norm_list = [True]
gpm_sample = 80
buffer_size_list = [50]
buffer_type_list = ['mean']
best_model_metric_list = ['macro_f1']
method_list = ['matrix']
ROGPM_threshold_list = [0.8]
sample_selection = 'random'


# FT
for dataset_path, use_sub_exp_ft in dataset_path_list:
    for best_model_metric in best_model_metric_list:
        program_run = r'python Baseline.py --baseline_mode FT --ft_load_base --cuda_id {} --network_architecture {} --epochs {} --batch_size {} --use_ft_pretrained_network --dataset_path {} --fix_norm_affine --best_model_metric {} {}'.format(
            cuda_id, network_architecture, epochs, batch_size, dataset_path,
            best_model_metric, '--use_sub_exp_ft' if use_sub_exp_ft else '')
        run_(program_run)


# JT
for dataset_path, use_sub_exp_ft in dataset_path_list:
    for best_model_metric in best_model_metric_list:
        program_run = r'python Baseline.py --baseline_mode JT --cuda_id {} --network_architecture {} --epochs {} --batch_size {} --use_ft_pretrained_network --dataset_path {} --fix_norm_affine --best_model_metric {} {}'.format(cuda_id, network_architecture, epochs, batch_size, dataset_path, best_model_metric, '--use_sub_exp_ft' if use_sub_exp_ft else '')
        run_(program_run)


# RO-GPM
for dataset_path, use_sub_exp_ft in dataset_path_list:
    for rogpm_threshold in ROGPM_threshold_list:
        for buffer_size in buffer_size_list:
            for buffer_type in buffer_type_list:
                for best_model_metric in best_model_metric_list:
                    program_run = r'python RO-GPM.py {} {} --cuda_id {} --network_architecture {} --epochs {} --batch_size {} --use_ft_pretrained_network --sample_selection_method {} --sample_selection_num {} 0 0 {} --dataset_path {} --threshold {} --buffer_size {} --buffer_type {} --best_model_metric {}'.format(
                        '--fix_norm_affine', '--use_sub_exp_ft' if use_sub_exp_ft else '', cuda_id, network_architecture, epochs, batch_size, sample_selection, gpm_sample,
                        '--select_sample_per_class', dataset_path, rogpm_threshold, buffer_size, buffer_type, best_model_metric)
                    run_(program_run)



