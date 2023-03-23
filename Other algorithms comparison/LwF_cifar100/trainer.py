import torch
import torchvision
from torchvision.models import vgg16 #用VGG的时候就用VGG，不用VGG的时候还可以改为Resnet
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
from model import PreResNet
# from model1 import VGG16
from cifar import Cifar100
# from exemplar import Exemplar
from copy import deepcopy #深度复制，用于复制模型
import torch.backends.cudnn as cudnn


#核心代码
class Trainer:
    def __init__(self, total_cls):
        cudnn.benchmark = True
        self.total_cls = total_cls
        # seen_cls帮助我们记住模型第几次增量，并且模型已经学习了几类
        self.seen_cls = 0
        self.dataset = Cifar100()
        self.model = PreResNet(32,total_cls).cuda()

        # 这里就是切换resnet模型和vgg模型
        # self.model = VGG16(total_cls)
        # print(self.model)
        # self.model = nn.DataParallel(self.model, device_ids=[0])
        self.input_transform= Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32,padding=4),
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])

        self.input_transform_eval= Compose([
                                ToTensor(),
                                Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])])
        
       # 显示模型参数
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("Solver total trainable parameters : ", total_params)

    # 测试用例，为了计算测试的准确率吧
    def test(self, testdata):
        print("test data number : ",len(testdata))
        # 我要固定不改变模型的任何东西
        self.model.eval()

        # 在测试之前全部清零
        count = 0
        correct = 0
        wrong = 0

        # 将测试集放入
        for i, (image, label) in enumerate(testdata):
            image = image.cuda()
            label = label.view(-1).cuda()

            # 图片放入模型中
            p = self.model(image)
            # 用argmax进行预测
            pred = p[:,:self.seen_cls].argmax(dim=-1)
            # 输出预测结果 Pred

            correct += sum(pred == label).item()
            wrong += sum(pred != label).item()
        acc = correct / (wrong + correct)
        print("Test Acc: {}".format(acc*100))
        self.model.train()
        # 模型可以训练了
        print("---------------------------------------------")
        return acc

    # 获得学习率
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # 开始训练了
    def train(self, batch_size, epoches, lr, max_size):
        total_cls = self.total_cls
        # 标准交叉熵
        criterion = nn.CrossEntropyLoss()
        # 因为每一次增量的时候，我们其实还是要保存上一次的模型
        # 科研中重要
        previous_model = None

        dataset = self.dataset
        test_xs = []
        test_ys = []

        test_accs = []
        # 0 1 2 3 4
        for inc_i in range(dataset.batch_num):  #batch_num是增量次数
            print(f"Incremental num : {inc_i}")
            # 我要开始取相关的已经定义好的数据集
            train, test = dataset.getNextClasses(inc_i)
            print(len(train), len(test))
            train_x, train_y = zip(*train)
            test_x, test_y = zip(*test)
            # 测试数据集，每次取数据集，我们是不是只能取5大类中的1类
            # 但是测试的时候，我们希望之前已经学习过的测试类一块被测试
            test_xs.extend(test_x)
            test_ys.extend(test_y)

            #可以被识别     
            train_data = DataLoader(BatchData(train_x, train_y, self.input_transform),
                        batch_size=batch_size, shuffle=True, drop_last=True)
            test_data = DataLoader(BatchData(test_xs, test_ys, self.input_transform_eval),
                        batch_size=batch_size, shuffle=False)

            # 优化器
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9,  weight_decay=2e-4)
            # scheduler = LambdaLR(optimizer, lr_lambda=adjust_cifar100)

            # 控制学习速率，每到70、140、210...的时候，学习速率要*0.1
            scheduler = StepLR(optimizer, step_size=70, gamma=0.1)

            # 代表模型要开始学了，并且学到了20类
            self.seen_cls = 20 + inc_i *20
            print("seen cls number : ", self.seen_cls)

            test_acc = []
            # 我要保存模型的参数了
            ckp_name = './checkpoint/{}_run_{}_iteration_{}_model.pth'.format(self.seen_cls-20, self.seen_cls, inc_i)
            if os.path.exists(ckp_name):
                # 如果有现成的，直接加载，如果没有，那么我重新训练
                self.model = torch.load(ckp_name)
            else:
                # 正式训练
                for epoch in range(epoches):
                    print("---"*50)
                    print("Epoch", epoch)
                    scheduler.step()
                    cur_lr = self.get_lr(optimizer)
                    print("Current Learning Rate : ", cur_lr)
                    self.model.train()
                    # 注意，这一块开始了
                    if inc_i > 0:
                        # 要用增量学习了
                        self.stage1_distill(train_data, criterion, optimizer)
                    else:
                        # 用正常的学习
                        self.stage1(train_data, criterion, optimizer)
                    # 输出测试结果
                    acc = self.test(test_data)
                # 保存模型 
                torch.save(self.model, ckp_name)
            # 复制模型
            self.previous_model = deepcopy(self.model)
            # 最终的准确率
            acc = self.test(test_data)
            test_acc.append(acc)
            test_accs.append(max(test_acc))
            print(test_accs)

    # 没啥用，作用就是【0 1 0 0 0 】【2】
    def get_one_hot(self, target,num_class):
        one_hot=torch.zeros(target.shape[0],num_class).cuda()
        one_hot=one_hot.scatter_(dim=1,index=target.long().view(-1,1),value=1.)
        return one_hot

    # 正常的学习
    def stage1(self, train_data, criterion, optimizer):
        print("Training ... ")
        losses = []
        for i, (image, label) in enumerate(tqdm(train_data)): # 显示学习进度条
            image = image.cuda()
            label = label.view(-1).cuda()
            p = self.model(image)
            # 显示交叉熵损失函数，以数字的形式
            loss = criterion(p[:,:self.seen_cls], label)
            optimizer.zero_grad() #优化
            # 回传
            loss.backward(retain_graph=True)
            optimizer.step()
            losses.append(loss.item())
        print("stage1 loss :", np.mean(losses))

    # 创新点LwF（CVPR）
    def stage1_distill(self, train_data, criterion, optimizer):
        print("Training ... ")
        distill_losses = []
        ce_losses = []
        T = 2
        # alpha = (self.seen_cls - 2)/ self.seen_cls 
        # 参数，这个参数你可以随意实验
        # print("classification proportion 1-alpha = ", 1-alpha)
        for i, (image, label) in enumerate(tqdm(train_data)):
            image = image.cuda()
            label = label.view(-1).cuda()
            # image = torch.from_numpy(np.concatenate((image, image, image), axis=-3))
            p = self.model(image)
            
            # 代表的下面的模型参数，不会传（参数不变）
            with torch.no_grad():
                pre_p = self.previous_model(image)
                # 上一次的模型
                pre_p = F.softmax(pre_p[:,:self.seen_cls-20]/T, dim=1)

            #蒸馏损失函数
            logp = F.log_softmax(p[:,:self.seen_cls-20]/T, dim=1)
            loss_soft_target = -torch.mean(torch.sum(pre_p * logp, dim=1))
            loss_hard_target = nn.CrossEntropyLoss()(p[:,self.seen_cls-20:self.seen_cls], label-self.seen_cls+20)
            # loss_hard_target = nn.CrossEntropyLoss()(p[:,self.seen_cls-20:self.seen_cls], label)
            loss = loss_soft_target +  loss_hard_target
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            distill_losses.append(loss_soft_target.item())
            ce_losses.append(loss_hard_target.item())
        print("stage1 distill loss :", np.mean(distill_losses), "ce loss :", np.mean(ce_losses))



