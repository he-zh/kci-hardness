import numpy as np
import yaml
import argparse
from kernel_selection import KernelSelection
from utils import get_error_mean_std
import json
import os
import torch
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Define a function to save intermediate results
def save_results(p_values, save_path):
    with open(save_path, 'w') as f:
        json.dump(p_values, f)

# Define a function to load saved results
def load_saved_results(save_path):
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            return json.load(f)
    return {}


def generate_save_path(config, ground_truth=None):
    dataset = config['data']['dataset']
    data_split_type = config['data']['data_split_type']
    fixed_size = config['data']['fixed_size']
    model_c = config['model']['model_c']
    model_ca = config['model']['model_ca']
    model_cb = config['model']['model_cb']
    gamma_c = config['model']['gamma_c']
    gamma_ca = config['model']['gamma_ca']
    gamma_cb = config['model']['gamma_cb']
    is_trainable_c = 'train' if config['model']['is_trainable_c'] else 'no_train'
    is_trainable_ca = 'train' if config['model']['is_trainable_ca'] else 'no_train'
    is_trainable_cb = 'train' if config['model']['is_trainable_cb'] else 'no_train'
    early_stopping = 'earlystop' if config['model'].get('early_stopping', False) else ''

    pval_approx_type = config['model']['pval_approx_type']
    
    dim = config['data'].get('dim', '')
    save_dir = os.path.join(
        config['output']['output_dir'], dataset, f"dim{dim}", 
        f"{data_split_type}_{fixed_size}/{is_trainable_c}_c_{model_c}_{gamma_c}-{is_trainable_ca}_ca_{model_ca}_{gamma_ca}-{is_trainable_cb}_cb_{model_cb}_{gamma_cb}-{early_stopping}_{pval_approx_type}"
    )
    if 'splitkci' in config:
        save_dir = os.path.join(save_dir, 'splitkci') if config['splitkci'] else save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ground_truth = config['data']['ground_truth']
    n_runs = config['experiment']['n_runs']

    file_name = f"{ground_truth}_{n_runs}runs.json"
    save_path = os.path.join(save_dir, file_name)
    return save_path

def get_train_test_sizes(dataset_size, data_split_type, fixed_size, train_proportion=0.8, **ignored):
    if data_split_type == 'no_split':
        return dataset_size, dataset_size
    elif data_split_type == 'fixed_train_size':
        train_size = fixed_size
    elif data_split_type == 'fixed_test_size':
        train_size = dataset_size - fixed_size
    elif data_split_type == 'proportional':
        train_size = int(train_proportion * dataset_size)
    else:
        raise NotImplementedError("The data split type is not implemented.")
    return train_size, dataset_size - train_size

def train_test_split(a, b, c, data_split_type, train_size, seed=0, **ignored):
    if data_split_type == 'no_split':
        return a, b, c, a, b, c

    indices = np.arange(a.shape[0])
    np.random.RandomState(seed=seed).shuffle(indices)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    a_train, b_train, c_train = a[train_indices], b[train_indices], c[train_indices]
    a_test, b_test, c_test = a[test_indices], b[test_indices], c[test_indices]
    return a_train, b_train, c_train, a_test, b_test, c_test


def main(config_file):
    config = load_config(config_file)

    for ground_truth in ['H0','H1']:
        save_path = generate_save_path(config, ground_truth)
        if os.path.exists(save_path):
            p_values = load_saved_results(save_path)
            p_values = {int(k): v for k, v in p_values.items()}
        else:
            p_values = {}

        for n_points in range(400, 1201, 100):
            train_size, test_size = get_train_test_sizes(dataset_size=n_points, **config['data'])
            for i in range(config['experiment']['n_runs']):
                if n_points in p_values and len(p_values[n_points]) >= i + 1:
                    continue
                a, b, c = getattr(data, config['data']['dataset'])(**config['data'], seed=i, n_points=n_points, 
                                                                   ground_truth=ground_truth,
                                                                   device=device)
                a_train, b_train, c_train, a_test, b_test, c_test = train_test_split(a, b, c, train_size=train_size, 
                                                                                     seed=i, **config['data'])

                model = KernelSelection(**config['model'], device=device)
                model.fit(a_train, b_train, c_train)
                pval = model.compute_p_value(a_test, b_test, c_test)
                if n_points not in p_values:
                    p_values[n_points] = []
                p_values[n_points].append(pval)
                if config['output']['save_results']: save_results(p_values, save_path)
            result_mean, var = get_error_mean_std(p_values[n_points], ground_truth, 0.05)
            print("==================================================================================")
            print(f"hypothesis: {ground_truth}, train nums: {train_size}, test nums: {test_size}, error mean: {result_mean}, std: {var}")
            print("==================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='config/1d_power_kci.yaml')
    args = parser.parse_args()
    main(args.config_file)
    
