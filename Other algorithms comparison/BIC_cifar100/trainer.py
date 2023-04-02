import torch
import torchvision
from torchvision.models import vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize, ToTensor, ToPILImage
from torch.optim.lr_scheduler import LambdaLR, StepLR

import numpy as np
import glob
import PIL.Image as Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import pickle
from dataset import BatchData
from model import PreResNet, BiasLayer
# from model1 import VGG16
from cifar import Cifar100
from exemplar import Exemplar
from copy import deepcopy
import torch.backends.cudnn as cudnn


class Trainer:
    def __init__(self, total_cls):
        cudnn.benchmark = True
        self.total_cls = total_cls
        self.seen_cls = 0
        self.dataset = Cifar100()
        self.model = PreResNet(32,total_cls).cuda()
        # self.model = VGG16(total_cls)
        # print(self.model)
        self.model = nn.DataParallel(self.model, device_ids=[0])
        # 定义五次（因为是五次增量学习）的参数（每次的参数都不一样） 
        # 偏置层，定义于model
        self.bias_layer1 = BiasLayer().cuda()
        self.bias_layer2 = BiasLayer().cuda()
        self.bias_layer3 = BiasLayer().cuda()
        self.bias_layer4 = BiasLayer().cuda()
        self.bias_layer5 = BiasLayer().cuda()
        self.bias_layers=[self.bias_layer1, self.bias_layer2, self.bias_layer3, self.bias_layer4, self.bias_layer5]
        self.input_transform= Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32,padding=4),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        self.input_transform_eval= Compose([
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)


    def test(self, testdata):
        print("test data number : ",len(testdata))
        self.model.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        print("---------------------------------------------")
        return acc

    def test_1(self, testdata):
        print("test data number : ",len(testdata))
        self.model.eval()
        # self.bias_forward.eval()
        count = 0
        correct = 0
        wrong = 0
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()
            with torch.no_grad():
                p = self.model(image)
                p = self.bias_forward(p)
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        # self.bias_forward.train()
        print("---------------------------------------------")
        return acc

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        criterion = nn.CrossEntropyLoss()
        exemplar = Exemplar(max_size, total_cls)

        previous_model = None

        dataset = self.dataset
        test_xs = []
        test_ys = []
        train_xs = []
        train_ys = []

        test_accs = []
        for inc_i in range(dataset.batch_num):
            print(f"Incremental num : {inc_i}")
            train, val, test = dataset.getNextClasses(inc_i)
            print(len(train), len(test))
            train_x, train_y = zip(*train)
            val_x, val_y = zip(*val)
            test_x, test_y = zip(*test)
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            train_xs, train_ys = exemplar.get_exemplar_train()
            train_xs.extend(train_x)
            train_xs.extend(val_x)
            train_ys.extend(train_y)
            train_ys.extend(val_y)

            train_data = DataLoader(BatchData(train_xs, train_ys, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)
            scheduler = StepLR(optimizer, step_size=70, gamma=0.1)

            bias_optimizer = optim.Adam(self.bias_layers[inc_i].parameters(), lr=0.001)
            exemplar.update(total_cls//dataset.batch_num, (train_x, train_y),(val_x, val_y))

            # self.seen_cls = 20 + inc_i *20
            self.seen_cls = exemplar.get_cur_cls()
            print("seen cls number : ", self.seen_cls)

            val_xs, val_ys = exemplar.get_exemplar_val()
            val_bias_data = DataLoader(BatchData(val_xs, val_ys, self.input_transform),
                        batch_size=100, shuffle=True, drop_last=False)

            test_acc = []
            ckp_name = './checkpoint/{}_run_{}_iteration_{}_model.pth'.format(self.seen_cls-20, self.seen_cls, inc_i)
            if os.path.exists(ckp_name):
                self.model = torch.load(ckp_name)
            else:
                for epoch in range(epoches):
                    print("---"*50)
                    print("Epoch", epoch)
                    scheduler.step()
                    cur_lr = self.get_lr(optimizer)
                    print("Current Learning Rate : ", cur_lr)
                    self.model.train()
                    # 增量学习
                    if inc_i > 0:
                        self.stage1_distill(train_data, criterion, optimizer)
                    else:
                        # 普通学习
                        self.stage1(train_data, criterion, optimizer)
                    acc = self.test(test_data)
                # 继续判断，如果使用增量学习的话 
                if inc_i > 0:
                    for epoch in range(epoches):
                        # bias_scheduler.step()
                        self.stage2(val_bias_data, criterion, bias_optimizer)
                        # BIC算法
                        if epoch % 50 == 0:
                            acc = self.test_1(test_data)
                            test_acc.append(acc)
                for i, layer in enumerate(self.bias_layers):
                    layer.printParam(i)
                torch.save(self.model, ckp_name)
            self.previous_model = deepcopy(self.model)
            if inc_i == 0:
                acc = self.test(test_data)
            else:
                acc = self.test_1(test_data)
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            print(test_accs)

    # def get_one_hot(self, target,num_class):
    #     one_hot=torch.zeros(target.shape[0],num_class).cuda()
    #     one_hot=one_hot.scatter_(dim=1,index=target.long().view(-1,1),value=1.)
    #     return one_hot

    # 这块是核心
    def bias_forward(self, input):
        in1 = input[:, :20]
        in2 = input[:, 20:40]
        in3 = input[:, 40:60]
        in4 = input[:, 60:80]
        in5 = input[:, 80:100]
        out1 = self.bias_layer1(in1)
        out2 = self.bias_layer2(in2)
        out3 = self.bias_layer3(in3)
        out4 = self.bias_layer4(in4)
        out5 = self.bias_layer5(in5)
        #与论文中的相对应
        # 连接
        return torch.cat([out1, out2, out3, out4, out5], dim = 1)

    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    def stage1_distill(self, train_data, criterion, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        alpha = (self.seen_cls - 20)/ self.seen_cls
        print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            p = self.bias_forward(p)
            with torch.no_grad():
                pre_p = self.previous_model(image)
                pre_p = self.bias_forward(pre_p)
                pre_p = F.softmax(pre_p[:,:self.seen_cls-20]/T, dim=1)
            logp = F.log_softmax(p[:,:self.seen_cls-20]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            # loss_hard_target = nn.CrossEntropyLoss()(p[:,self.seen_cls-20:self.seen_cls], label-self.seen_cls+20)
            loss_hard_target = nn.CrossEntropyLoss()(p[:,:self.seen_cls], label)
            loss = loss_soft_target * (alpha)+ loss_hard_target *(1-alpha)
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))

    # BIC
    def stage2(self, val_bias_data, criterion, optimizer):
        print("Evaluating ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(val_bias_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            with torch.no_grad():
                p = self.model(image)
            # 采用自己的BIC算法进行训练 在正常训练之后，加入一个BIC模块
            p = self.bias_forward(p)
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("stage2 loss :", np.mean(losses))

