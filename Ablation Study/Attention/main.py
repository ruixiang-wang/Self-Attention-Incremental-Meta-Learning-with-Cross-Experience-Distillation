# Author: Ghada Sokar  et al.
# This is the official implementation of the paper Self-Attention Meta-Learner for Continual Learning at AAMAS 2021
# if you use part of this code, please cite the following article 
# @inproceedings{sokar2021selfattention,
#   title={Self-Attention Meta-Learner for Continual Learning},
#   author={Ghada Sokar and Decebal Constantin Mocanu and Mykola Pechenizkiy},
#   booktitle={20th International Conference on Autonomous Agents and Multiagent Systems, AAMAS 2021},
#   year={2021},
#   organization={International Foundation for Autonomous Agents and Multiagent Systems (IFAAMAS)}
# }

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import specific_subnetwork
import shared_subnetwork
import utils

desc = "Pytorch implementation of Self-Attention Meta-Learner for CL (SAM) on the split MNIST benchmark"
parser = argparse.ArgumentParser(description=desc)
# training paramaeters
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--optimizer', type=str, default='sgd', help='The optimizer to use. Default: sgd. Options: sgd, adam.')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--l2', type=float, default=0.005)
parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                    help='How many batches to wait before logging training status')

parser.add_argument('--max_threads', type=int, default=0, help='How many threads to use for data loading.')
parser.add_argument('--seed', type=int, default=4, metavar='S')
parser.add_argument('--meta_model_path', type=str, default='./meta_learner_model/shared_sub_network.pt', help='Path of the self-attention meta-learner')


def evaluate(shared_subnetwork_model, specific_subnetwork, criterion, device, test_loader, is_test_set=False):
    specific_subnetwork.eval()
    test_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            selected_representation = shared_subnetwork_model(data)
            output = specific_subnetwork(selected_representation)
            test_loss += criterion(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

    test_loss /= float(n)

    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))              
    return 100. * correct / float(n)

def evaluate_tasks(shared_subnetwork_model, specific_models, task_id, device, task_test_loader, num_classes_per_task, is_test_set=False):
    correct_task = 0
    correct_class = 0    
    n = 0
    with torch.no_grad():
        for data, target in task_test_loader:
            data, target = data.to(device), target.to(device, dtype=torch.int64)
            target.data = target.data + task_id*num_classes_per_task
            n += target.shape[0]

            selected_representation = shared_subnetwork_model(data)
            for i in range (len(specific_models)):
                specific_models[i].eval()
                model_output = specific_models[i](selected_representation)
                if i ==0:
                   concat_op = model_output
                else:   
                   concat_op = torch.cat((concat_op,model_output),1) 

            pred_class = concat_op.argmax(dim=1, keepdim=True)
            correct_class += pred_class.eq(target.view_as(pred_class)).sum().item()
            correct_task += (pred_class.data//num_classes_per_task==task_id).sum().item()

    print('\n{} of task_id: {} Accuracy of selecting the correct Task: {:.3f} Accuracy of correct target: {}/{} ({:.3f}%) \n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',task_id,
         100. * correct_task / float(n), correct_class, n, 100*correct_class/float(n)))  

    return 100. * correct_class / float(n), 100. * correct_task / float(n)


def train(args, shared_subnetwork_model, specific_model, epochs, log_interval, device, train_loader, test_loader, batch_size):
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, specific_model.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=True)

    elif args.optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, specific_model.parameters()), lr=args.lr, betas=(0.9, .999))

    for epoch in range(epochs):
        specific_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device, dtype=torch.int64)  
            optimizer.zero_grad()
            with torch.no_grad():
                selected_representation = shared_subnetwork_model(data)
            outputs = specific_model(selected_representation)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item())) 

        val_acc = evaluate(shared_subnetwork_model, specific_model, criterion, device, test_loader, is_test_set=True)

    return val_acc

def run_CL_tasks(args):

    #---------------------------------------------------------------------------------------------------------#
    #---------- DATA --------------#
    #------------------------------#
    task_labels = [[0,1], [2,3], [4,5], [6,7], [8,9]]
    num_classes_per_task = 2
    train_dataset, test_dataset = utils.task_construction(task_labels)  

    #----------------------------------------------------------------------------------------------------------#
    #------------------------------ META LEARNER MODEL --------------------------------------------------------#
    #---We changed MAML implementation from https://github.com/dragen1860/MAML-Pytorch ------------------------# 
    #---to include the attention module and work for MLP-------------------------------------------------------#
    #---We trained the self-attention meta-learner model, the trained model can be found in meta_learner_model-#
    #----------------------------------------------------------------------------------------------------------#
    # load meta learned params for shared sub-network
    print('meta_learner_path: '+ args.meta_model_path)
    use_cuda =  torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used: '+ str(device))
    shared_SAM_model = torch.load(args.meta_model_path, map_location=torch.device(device))
    SAM = shared_subnetwork.shared_subnetwork(shared_SAM_model.vars, shared_SAM_model.vars_bn).to(device)
    
    #---------------------------------------------------------------------------------------------------------#
    #---------- Continual Learning a sequence of tasks -------------------------------#
    #---------------------------------------------------------------------------------#
    specific_models = []
    train_loaders = []
    test_loaders = []
    total_acc_conditioned = 0
    all_acc = []
    for i in range(len(task_labels)):
        all_acc.append([])
        train_loaders.append(utils.get_task_train_loader(train_dataset[i], args.batch_size, args.max_threads))
        test_loaders.append(utils.get_task_test_loader(test_dataset[i], args.batch_size))
        specific_models.append(specific_subnetwork.MLP(num_classes_per_task).to(device))
        total_acc_conditioned += train(args, SAM, specific_models[i], args.epochs, args.log_interval, device, train_loaders[i], test_loaders[i], args.batch_size)

    #------------------------------------------------------------------------------------------#
    #----- Performance on each task after training the whole sequence -----#
    #----------------------------------------------------------------------#
    total_acc_agnostic=0
    total_correct_task=0
    for i in range(len(task_labels)):
        class_acc, task_acc = evaluate_tasks(SAM,specific_models, i, device, test_loaders[i], num_classes_per_task, True)
        total_acc_agnostic += class_acc
        total_correct_task += task_acc
    
    print('Average  Accuracy in task agnostic inference:  {:.3f}'.format(total_acc_agnostic/len(task_labels)))
    print('Average  Accuracy in task conditioned inference: {:.3f}'.format(total_acc_conditioned/len(task_labels)))
    print('Average  Accuracy of identifying the correct task: {:.3f}'.format(total_correct_task/len(task_labels)))

if __name__ == '__main__':
    # load input arguments
    args = parser.parse_args()
    print(args)

    # set seed 
    os.environ['PYTHONHASHSEED']=str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available() 
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # learning a sequence of continual learning tasks
    run_CL_tasks(args)