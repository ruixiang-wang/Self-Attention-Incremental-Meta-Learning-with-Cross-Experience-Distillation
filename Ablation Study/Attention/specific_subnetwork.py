# Author: Ghada Sokar  et al.
# This is the official implementation of the paper Self-Attention Meta-Learner for Continual Learning at AAMAS 2021

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self,num_classes):
        super(MLP, self).__init__()
        # model arch
        self.num_classes = num_classes
        self.featuremap_layers_size = [400,400]
        
        # specific sub-network
        self.classifier = nn.Sequential(
            nn.Linear(self.featuremap_layers_size[-1], self.num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x    
