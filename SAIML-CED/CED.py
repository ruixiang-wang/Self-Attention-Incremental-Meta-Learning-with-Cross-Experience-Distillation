from __future__ import print_function
import pickle
from net import *
import os
import dataloader as data
from torch.utils.data import RandomSampler

def compute_kl_divergence(new_model, old_model, new_data, old_data):
    new_model.eval()
    old_model.eval()

    if torch.cuda.is_available():
        new_data = new_data.cuda()
        old_data = old_data.cuda()

    # Compute the output distributions of the new and old model on the new data
    new_model_new_data_output = F.softmax(new_model(new_data)[0], dim=1)
    old_model_new_data_output = F.softmax(old_model(new_data)[0], dim=1)

    # Compute the KL divergence between the new and old model distributions on the new data
    kl_div_new_data = F.kl_div(F.log_softmax(new_model_new_data_output, dim=1), old_model_new_data_output,
                               reduction='batchmean')

    # Compute the output distributions of the new and old model on the old data
    new_model_old_data_output = F.softmax(new_model(old_data)[0], dim=1)
    old_model_old_data_output = F.softmax(old_model(old_data)[0], dim=1)

    # Compute the KL divergence between the new and old model distributions on the old data
    kl_div_old_data = F.kl_div(F.log_softmax(new_model_old_data_output, dim=1), old_model_old_data_output,
                               reduction='batchmean')

    # Add the two KL divergences
    total_kl_div = 0.5 * kl_div_new_data + 0.5 * kl_div_old_data

    return total_kl_div


