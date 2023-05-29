
import os
import torch
import torch.optim as optim
import time
import pickle
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import copy
from net.ResNet import *
import random
from radam import *
from CEDLoss import cross_generalization_distillation



class Learner():
    def __init__(self, learnerModel, arguments, trainDataLoader, testDataLoader, useGPU):
        self.learnerModel = learnerModel
        self.bestLearnerModel = learnerModel
        self.arguments = arguments
        self.trainDataLoader = trainDataLoader
        self.useGPU = useGPU
        self.learnerState = {key: value for key, value in self.arguments.__dict__.items() if
                             not key.startswith('__') and not callable(key)}
        self.bestAcc = 0
        self.testDataLoader = testDataLoader
        self.testLoss = 0.0
        self.testAcc = 0.0
        self.trainLoss, self.trainAcc = 0.0, 0.0

        modelParams = []
        for name, param in self.learnerModel.named_parameters():
            modelParams.append(param)
            param.requires_grad = True

        if (self.arguments.optimizer == "radam"):
            self.learnerOptimizer = RAdam(modelParams, lr=self.arguments.lr, betas=(0.9, 0.999), weight_decay=0)
        elif (self.arguments.optimizer == "adam"):
            self.learnerOptimizer = optim.Adam(self.learnerModel.parameters(), lr=self.arguments.lr, betas=(0.9, 0.999),
                                               eps=1e-08, weight_decay=0.0, amsgrad=False)
        elif (self.arguments.optimizer == "sgd"):
            self.learnerOptimizer = optim.SGD(modelParams, lr=self.arguments.lr, momentum=0.9, weight_decay=0.001)

    def learn(self):
        logWriter = Logger(os.path.join(self.arguments.checkpoint, 'session_' + str(self.arguments.sess) + '_log.txt'),
                           title=self.title)
        logWriter.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc'])

        for epochNum in range(0, self.arguments.epochs):
            self.adjust_learning_rate(epochNum)
            print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (
            epochNum + 1, self.arguments.epochs, self.learnerState['lr'], self.arguments.sess))

            self.train(self.learnerModel, epochNum)
            self.test(self.learnerModel)

            logWriter.append(
                [self.learnerState['lr'], self.trainLoss, self.testLoss, self.trainAcc, self.testAcc, self.bestAcc])

            is_best = self.testAcc > self.bestAcc
            if (is_best and epochNum > self.arguments.epochs - 10):
                self.bestLearnerModel = copy.deepcopy(self.learnerModel)

            self.bestAcc = max(self.testAcc, self.bestAcc)
            if (epochNum == self.arguments.epochs - 1):
                self.save_checkpoint(self.bestLearnerModel.state_dict(), True, checkpoint=self.arguments.savepoint,
                                     filename='session_' + str(self.arguments.sess) + '_model_best.pth.tar')
        self.learnerModel = copy.deepcopy(self.bestLearnerModel)

        logWriter.close()
        logWriter.plot()
        savefig(os.path.join(self.arguments.checkpoint, 'log.eps'))

        print('Best acc:')
        print(self.bestAcc)

    def train(self, learnerModel, epochNum):
        learnerModel.train()

        avgBatchTime = AverageMeter()
        avgDataTime = AverageMeter()
        avgLosses = AverageMeter()
        avgTop1 = AverageMeter()
        avgTop5 = AverageMeter()
        timeEnd = time.time()

        targetClasses = self.arguments.class_per_task * (1 + self.arguments.sess)

        for batchIndex, (inputData, targetData) in enumerate(self.trainDataLoader):
            avgDataTime.update(time.time() - timeEnd)
            learnerSessions = []

            targetDataOneHot = torch.FloatTensor(inputData.shape[0], targetClasses)
            targetDataOneHot.zero_()
            targetDataOneHot.scatter_(1, targetData[:, None], 1)

            if self.useGPU:
                inputData, targetDataOneHot, targetData = inputData.cuda(), targetDataOneHot.cuda(), targetData.cuda()
            inputData, targetDataOneHot, targetData = torch.autograd.Variable(inputData), torch.autograd.Variable(
                targetDataOneHot), torch.autograd.Variable(targetData)

            reptileGrads = {}
            npTargetData = targetData.detach().cpu().numpy()
            numUpdates = 0

            outputData, _ = learnerModel(inputData)

            learnerModelBase = copy.deepcopy(learnerModel)
            for taskIndex in range(1 + self.arguments.sess):
                dataIndex = np.where((npTargetData >= taskIndex * self.arguments.class_per_task) & (
                            npTargetData < (taskIndex + 1) * self.arguments.class_per_task))[0]

                if (len(dataIndex) > 0):
                    learnerSessions.append([taskIndex, 0])
                    for i, (param, baseParam) in enumerate(
                            zip(learnerModel.parameters(), learnerModelBase.parameters())):
                        param = copy.deepcopy(baseParam)

                    classInputData = inputData[dataIndex]
                    classTargetDataOneHot = targetDataOneHot[dataIndex]
                    classTargetData = targetData[dataIndex]

                    for kr in range(self.arguments.r):
                        _, classOutputData = learnerModel(classInputData)

                        classTargetLoss = classTargetDataOneHot.clone()
                        classPredictedLoss = classOutputData.clone()

                        loss = cross_generalization_distillation(learnerModel, learnerModelBase, classInputData,
                                                                 classTargetData, classPredictedLoss, self.arguments, μ,
                                                                 φ)

                        self.learnerOptimizer.zero_grad()
                        loss.backward()
                        self.learnerOptimizer.step()

                    for i, param in enumerate(learnerModel.parameters()):
                        if (numUpdates == 0):
                            reptileGrads[i] = [param.data]
                        else:
                            reptileGrads[i].append(param.data)
                    numUpdates += 1

            for i, (param, baseParam) in enumerate(zip(learnerModel.parameters(), learnerModelBase.parameters())):
                alpha = np.exp(-self.arguments.beta * ((1.0 * self.arguments.sess) / self.arguments.num_task))
                param.data = torch.mean(torch.stack(reptileGrads[i]), 0) * (alpha) + (1 - alpha) * baseParam.data

            prec1, prec5 = accuracy(output=learnerModel(inputData).data[:, 0:targetClasses],
                                    target=targetData.cuda().data, topk=(1, 1))
            avgLosses.update(loss.item(), inputData.size(0))
            avgTop1.update(prec1.item(), inputData.size(0))
            avgTop5.update(prec5.item(), inputData.size(0))

            avgBatchTime.update(time.time() - timeEnd)
            timeEnd = time.time()

            self.trainLoss, self.trainAcc = avgLosses.avg, avgTop1.avg

        def test(self):
            self.model.eval()
            losses = AverageMeter()
            top1 = AverageMeter()

            for batch_idx, (inputs, targets) in enumerate(self.dataLoader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
                outputs, _ = self.model(inputs)

                loss = cross_generalization_distillation(
                    self.model, model_old, exp_data_query, exp_proto, cls_loss, self.args, μ, φ)

                losses.update(loss.item(), inputs.size(0))
                prec1, _ = accuracy(outputs.data, targets.data, topk=(1, 1))
                top1.update(prec1.item(), inputs.size(0))

            self.test_loss = losses.avg
            self.test_acc = top1.avg

    def meta_test(self, memory, inc_dataset):
        self.model.eval()
        base_model = copy.deepcopy(self.model)
        meta_task_test_list = {}

        for task_idx in range(self.args.sess + 1):
            memory_data, memory_target = memory
            memory_data = np.array(memory_data, dtype="int32")
            memory_target = np.array(memory_target, dtype="int32")

            mem_idx = np.where((memory_target >= task_idx * self.args.class_per_task) & (
                        memory_target < (task_idx + 1) * self.args.class_per_task))[0]
            meta_memory_data = memory_data[mem_idx]
            meta_memory_target = memory_target[mem_idx]
            meta_model = copy.deepcopy(base_model)

            meta_loader = inc_dataset.get_custom_loader_idx(meta_memory_data, mode="train", batch_size=64)

            meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                                        weight_decay=0.0, amsgrad=False)

            meta_model.train()
            ai = self.args.class_per_task * task_idx
            bi = self.args.class_per_task * (task_idx + 1)

            if (self.args.sess != 0):
                for ep in range(1):
                    for batch_idx, (inputs, targets) in enumerate(meta_loader):
                        if self.use_cuda:
                            inputs, targets = inputs.cuda(), targets.cuda()
                        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                        _, outputs = meta_model(inputs)
                        loss = cross_generalization_distillation(self.model, model_old, exp_data_query, exp_proto,
                                                                 cls_loss, self.args, μ, φ)

                        meta_optimizer.zero_grad()
                        loss.backward()
                        meta_optimizer.step()

            meta_model.eval()
            for cl in range(self.args.class_per_task):
                class_idx = cl + self.args.class_per_task * task_idx
                loader = inc_dataset.get_custom_loader_class([class_idx], mode="test", batch_size=10)

                for batch_idx, (inputs, targets) in enumerate(loader):
                    if self.use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                    _, outputs = meta_model(inputs)

                    pred = torch.argmax(outputs[:, ai:bi], 1, keepdim=False)
                    pred = pred.view(1, -1)

                    correct_k = float(
                        torch.sum(pred.eq(targets.view(1, -1).expand_as(pred)).view(-1)).detach().cpu().numpy())
                    if (correct_k == 1):
                        if (pred[0][0] in meta_task_test_list.keys()):
                            meta_task_test_list[pred[0][0]] += 1
                        else:
                            meta_task_test_list[pred[0][0]] = 1
            meta_model.eval()
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

                _, outputs = meta_model(inputs)
                task_ids = outputs

                task_ids = task_ids.detach().cpu()
                outputs = outputs.detach().cpu()

                bs = inputs.size()[0]
                for i, t in enumerate(list(range(bs))):
                    j = batch_idx * self.args.test_batch + i
                    output_base_max = []
                    for si in range(self.args.sess + 1):
                        sj = outputs[i][si * self.args.class_per_task:(si + 1) * self.args.class_per_task]
                        sq = torch.max(sj)
                        output_base_max.append(sq)

                    task_argmax = np.argsort(outputs[i][ai:bi])[-5:]
                    task_max = outputs[i][ai:bi][task_argmax]

                    if (j not in meta_task_test_list.keys()):
                        meta_task_test_list[j] = [[task_argmax, task_max, output_base_max, targets[i]]]
                    else:
                        meta_task_test_list[j].append([task_argmax, task_max, output_base_max, targets[i]])
            del meta_model
        with open(self.args.savepoint + "/meta_task_test_list_" + str(task_idx) + ".pickle", 'wb') as handle:
            pickle.dump(meta_task_test_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_experience(self, experience, new_exp, seed=1):
        random.seed(seed)
        experience_per_task = self.args.memory // ((self.args.sess + 1) * self.args.class_per_task)
        self._data_exp, self._targets_exp = np.array([]), np.array([])
        mu = 1

        if (experience is not None):
            data_exp, targets_exp = experience
            data_exp = np.array(data_exp, dtype="int32")
            targets_exp = np.array(targets_exp, dtype="int32")
            for class_idx in range(self.args.class_per_task * (self.args.sess)):
                idx = np.where(targets_exp == class_idx)[0][:experience_per_task]
                self._data_exp = np.concatenate([self._data_exp, np.tile(data_exp[idx], (mu,))])
                self._targets_exp = np.concatenate([self._targets_exp, np.tile(targets_exp[idx], (mu,))])

        new_indices, new_targets = new_exp

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task * (self.args.sess),
                               self.args.class_per_task * (1 + self.args.sess)):
            idx = np.where(new_targets == class_idx)[0][:experience_per_task]
            self._data_exp = np.concatenate([self._data_exp, np.tile(new_indices[idx], (mu,))])
            self._targets_exp = np.concatenate([self._targets_exp, np.tile(new_targets[idx], (mu,))])

        print(len(self._data_exp))
        return list(self._data_exp.astype("int32")), list(self._targets_exp.astype("int32"))

    def store_checkpoint(self, state, is_best, checkpoint_dir, filename):
        if is_best:
            torch.save(state, os.path.join(checkpoint_dir, filename))

    def modify_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']
