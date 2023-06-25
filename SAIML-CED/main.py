from __future__ import print_function
import os
import time
import pickle
from net import *
from learner import Learner
import dataloader as data

class args:

    checkpoint = "results/cifar100/"
    savepoint = "models/" + "/".join(checkpoint.split("/")[1:])
    data_path = "../Datasets/CIFAR100/"
    num_class = 100
    class_per_task = 10
    num_task = 10
    test_samples_per_class = 100
    dataset = "cifar100"
    optimizer = "radam"
    
    epochs = 68
    lr = 0.01
    train_batch = 128
    test_batch = 100
    workers = 16
    sess = 0
    schedule = [20,40,60]
    gamma = 0.2
    random_classes = False
    validation = 0
    memory = 2000
    mu = 1
    beta = 1.0
    r = 2

use_cuda = torch.cuda.is_available()
seed = 2023
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed_all(seed)

def main():

    model = BasicNet1(args, 0).cuda()
    print('  Total params: %.2fM ' % (sum(p.numel() for p in model.parameters())/1000000.0))

    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)
    if not os.path.isdir(args.savepoint):
        os.makedirs(args.savepoint)
    inc_dataset = data.IncrementalDataset(
                        dataset_name=args.dataset,
                        args = args,
                        random_order=args.random_classes,
                        shuffle=True,
                        seed=1,
                        batch_size=args.train_batch,
                        workers=args.workers,
                        validation_split=args.validation,
                        increment=args.class_per_task,
                    )
        
    start_sess = 0
    memory = None
    
    for ses in range(start_sess, args.num_task):
        args.sess=ses 
        
        if(ses==0):
            torch.save(model.state_dict(), os.path.join(args.savepoint, 'base_model.pth.tar'))
            
        if(start_sess==ses and start_sess!=0): 
            inc_dataset._current_task = ses
            with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess-1)+".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing
            args.sample_per_task_testing = sample_per_task_testing
        
        if ses>0: 
            path_model=os.path.join(args.savepoint, 'session_'+str(ses-1) + '_model_best.pth.tar')  
            prev_best=torch.load(path_model)
            model.load_state_dict(prev_best)
            
            with open(args.savepoint + "/memory_"+str(args.sess-1)+".pickle", 'rb') as handle:
                memory = pickle.load(handle)
            
        task_info, train_loader, val_loader, test_loader, for_memory = inc_dataset.new_task(memory)
        print(task_info)
        print(inc_dataset.sample_per_task_testing)
        args.sample_per_task_testing = inc_dataset.sample_per_task_testing

        main_learner=Learner(model=model,args=args,trainloader=train_loader, testloader=test_loader, use_cuda=use_cuda, for_memory=for_memory)
        
        main_learner.learn()
        memory = inc_dataset.get_memory(memory, for_memory)       
        
        acc_task = main_learner.meta_test(main_learner.best_model, memory, inc_dataset)

        with open(args.savepoint + "/memory_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(args.savepoint + "/acc_task_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(args.savepoint + "/sample_per_task_testing_"+str(args.sess)+".pickle", 'wb') as handle:
            pickle.dump(args.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        time.sleep(10)
if __name__ == '__main__':
    main()
