# Author: Ghada Sokar  et al.
# This is the official implementation of the paper Self-Attention Meta-Learner for Continual Learning at AAMAS 2021

import torch
import torch.nn as nn
import torch.nn.functional as F

class shared_subnetwork(nn.Module):
    def __init__(self,vars,vars_bn):
        super(shared_subnetwork, self).__init__()
        self.input_size = 784
        self.featuremap_layers_size = [400,400]
        reduction=10
        
        #--- shared sub-network---#
        self.linear1=nn.Linear(self.input_size, self.featuremap_layers_size[0])
        self.linear1.weight.requires_grad = False
        self.linear1.bias.requires_grad = False
        # attention module
        self.se1_l1 = nn.Linear(self.featuremap_layers_size[0], self.featuremap_layers_size[0] // reduction, bias=False)
        self.se1_l1.weight.requires_grad = False
        self.se1_l2 = nn.Linear(self.featuremap_layers_size[0] // reduction, self.featuremap_layers_size[0], bias=False)
        self.se1_l2.weight.requires_grad = False

        self.linear2 = nn.Linear(self.featuremap_layers_size[0], self.featuremap_layers_size[1])
        self.linear2.weight.requires_grad = False
        self.linear2.bias.requires_grad = False
        # attention module
        self.se2_l1 = nn.Linear(self.featuremap_layers_size[1], self.featuremap_layers_size[1] // reduction, bias=False)
        self.se2_l1.weight.requires_grad = False
        self.se2_l2 = nn.Linear(self.featuremap_layers_size[1] // reduction, self.featuremap_layers_size[1], bias=False)
        self.se2_l2.weight.requires_grad = False
        
        # load trained weights from self-attention meta-learner that is trained on Omniglot dataset
        with torch.no_grad():
            self.linear1.weight.copy_(vars[0])
            self.linear1.bias.copy_(vars[1])
            self.se1_l1.weight.copy_(vars[4])
            self.se1_l2.weight.copy_(vars[5])
            
            self.linear2.weight.copy_(vars[6])
            self.linear2.bias.copy_(vars[7])
            self.se2_l1.weight.copy_(vars[10])
            self.se2_l2.weight.copy_(vars[11])

    def forward(self,x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x), inplace=True)
        y = F.relu(self.se1_l1(x),inplace=True)
        y = torch.sigmoid(self.se1_l2(y))
        x = x * y.expand_as(x)

        x = F.relu(self.linear2(x), inplace=True)
        y = F.relu(self.se2_l1(x),inplace=True)
        y = torch.sigmoid(self.se2_l2(y))
        x = x * y.expand_as(x)
        return x        