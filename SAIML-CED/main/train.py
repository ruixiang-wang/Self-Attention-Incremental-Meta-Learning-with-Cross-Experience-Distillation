import argparse
import os
import shutil
import time
import pickle
import torch
import pdb
import numpy as np
import copy
import torch
import sys
import random
import collections

from net.BasicNet1 import *
from Learner import Learner
import dataset.inc_dataloader as data
from radam import *


useGPU = torch.cuda.is_available()
randSeed = random.randint(1, 10000)
randSeed = 7572
random.seed(randSeed)
np.random.seed(randSeed)
torch.manual_seed(randSeed)
if useGPU:
    torch.cuda.manual_seed_all(randSeed)


def execute():
    netModel = BasicNet1(arguments, 0).cuda()

    print('  Total params: %.2fM ' % (sum(p.numel() for p in netModel.parameters()) / 1000000.0))

    if not os.path.isdir(arguments.checkpoint):
        mkdir_p(arguments.checkpoint)
    if not os.path.isdir(arguments.savepoint):
        mkdir_p(arguments.savepoint)
    np.save(arguments.checkpoint + "/seed.npy", randSeed)
    try:
        shutil.copy2('train_cifar.py', arguments.checkpoint)
        shutil.copy2('learner_task_itaml.py', arguments.checkpoint)
    except:
        pass
    incrDataset = data.IncrementalDataset(
        dataset_name=arguments.dataset,
        args=arguments,
        random_order=arguments.random_classes,
        shuffle=True,
        seed=1,
        batch_size=arguments.train_batch,
        workers=arguments.workers,
        validation_split=arguments.validation,
        increment=arguments.class_per_task,
    )

    startSession = int(sys.argv[1])
    memoryData = None

    for sessionNum in range(startSession, arguments.num_task):
        arguments.sess = sessionNum

        if (sessionNum == 0):
            torch.save(netModel.state_dict(), os.path.join(arguments.savepoint, 'base_model.pth.tar'))
            maskDict = {}

        if (startSession == sessionNum and startSession != 0):
            incrDataset._current_task = sessionNum
            with open(arguments.savepoint + "/sample_per_task_testing_" + str(arguments.sess - 1) + ".pickle",
                      'rb') as fileHandle:
                samplePerTaskTesting = pickle.load(fileHandle)
            incrDataset.sample_per_task_testing = samplePerTaskTesting
            arguments.sample_per_task_testing = samplePerTaskTesting

        if sessionNum > 0:
            pathModel = os.path.join(arguments.savepoint, 'session_' + str(sessionNum - 1) + '_model_best.pth.tar')
            previousBest = torch.load(pathModel)
            netModel.load_state_dict(previousBest)

            with open(arguments.savepoint + "/memory_" + str(arguments.sess - 1) + ".pickle", 'rb') as fileHandle:
                memoryData = pickle.load(fileHandle)

        taskData, trainLoader, valLoader, testLoader, forMemory = incrDataset.new_task(memoryData)
        print(taskData)
        print(incrDataset.sample_per_task_testing)
        arguments.sample_per_task_testing = incrDataset.sample_per_task_testing

        mainLearner = Learner(model=netModel, args=arguments, trainloader=trainLoader, testloader=testLoader,
                              use_cuda=useGPU)

        mainLearner.learn()
        memoryData = incrDataset.get_memory(memoryData, forMemory)

        taskAccuracy = mainLearner.meta_test(mainLearner.best_model, memoryData, incrDataset)

        with open(arguments.savepoint + "/memory_" + str(arguments.sess) + ".pickle", 'wb') as fileHandle:
            pickle.dump(memoryData, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(arguments.savepoint + "/acc_task_" + str(arguments.sess) + ".pickle", 'wb') as fileHandle:
            pickle.dump(taskAccuracy, fileHandle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/sample_per_task_testing_" + str(args.sess) + ".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)

        time.sleep(10)
if __name__ == '__main__':
    main()
