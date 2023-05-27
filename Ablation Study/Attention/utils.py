# Author: Ghada Sokar  et al.
# This is the official implementation of the paper Self-Attention Meta-Learner for Continual Learning at AAMAS 2021

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import copy

def get_task_train_loader(train_dataset, batch_size, max_threads=0):
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size,
    num_workers=max_threads,
    pin_memory=True, shuffle=True)
    return train_loader

def get_task_test_loader(test_dataset, test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    return test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    return full_dataset, test_dataset

def task_construction(task_labels):
    full_dataset,test_dataset = load_data()
    train_dataset = split_dataset_by_labels(full_dataset, task_labels)
    test_dataset = split_dataset_by_labels(test_dataset, task_labels)
    return train_dataset, test_dataset

def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    task_idx = 0
    for labels in task_labels:
        idx = np.in1d(dataset.targets, labels)
        splited_dataset = copy.deepcopy(dataset)
        splited_dataset.targets = splited_dataset.targets[idx]-task_idx*2
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
        task_idx += 1
    return datasets
