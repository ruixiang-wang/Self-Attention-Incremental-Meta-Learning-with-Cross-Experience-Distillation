import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
from torchvision import datasets, transforms

class SubsetRandomSampler(Sampler):

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if(self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        args,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.
    ):
        self.dataset_name = dataset_name.lower().strip()
        datasets = _get_datasets()
        self.train_transforms = datasets[0].train_transforms 
        self.common_transforms = datasets[0].common_transforms
        self.meta_transforms = datasets[0].train_transforms
        self.args = args
        
        self._setup_data(
            datasets,
            args.data_path,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )
        

        self._current_task = 0
        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self.sample_per_task_testing = {}
    @property
    def n_tasks(self):
        return len(self.increments)
    
    def get_same_index(self, target, label, mode="train", memory=None):
        label_indices = []
        label_targets = []

        for i in range(len(target)):
            if int(target[i]) in label:
                label_indices.append(i)
                label_targets.append(target[i])
        for_memory = (label_indices.copy(),label_targets.copy())

        if memory is not None:
            memory_indices, memory_targets = memory
            memory_indices2 = np.tile(memory_indices, (self.args.mu,))
            all_indices = np.concatenate([memory_indices2,label_indices])
        else:
            all_indices = label_indices
            
        return all_indices, for_memory
    
    def get_same_index_test_chunk(self, target, label, mode="test", memory=None):
        label_indices = []
        label_targets = []
        
        np_target = np.array(target, dtype="uint32")
        np_indices = np.array(list(range(len(target))), dtype="uint32")

        for t in range(len(label)//self.args.class_per_task):
            task_idx = []
            for class_id in label[t*self.args.class_per_task: (t+1)*self.args.class_per_task]:
                idx = np.where(np_target==class_id)[0]
                task_idx.extend(list(idx.ravel()))
            task_idx = np.array(task_idx, dtype="uint32")
            task_idx.ravel()
            random.shuffle(task_idx)

            label_indices.extend(list(np_indices[task_idx]))
            label_targets.extend(list(np_target[task_idx]))
            if(t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_idx)
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets

    def new_task(self, memory=None):
        print(self._current_task)
        print(self.increments)
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        
        train_indices, for_memory = self.get_same_index(self.train_dataset.targets, list(range(min_class, max_class)), mode="train", memory=memory)
        test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets, list(range(max_class)), mode="test")

        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,shuffle=False,num_workers=16, sampler=SubsetRandomSampler(train_indices, True))
        self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,shuffle=False,num_workers=16, sampler=SubsetRandomSampler(test_indices, False))

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indices),
            "n_test_data": len(test_indices)
        }

        self._current_task += 1
        return task_info, self.train_data_loader, self.test_data_loader, self.test_data_loader, for_memory
    
    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):
     
        if(mode=="train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, True))
        else: 
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader

    def get_custom_loader_class(self, class_id, mode="train", batch_size=10, shuffle=False):
        
        if(mode=="train"):
            train_indices, for_memory = self.get_same_index(self.train_dataset.targets, class_id, mode="train", memory=None)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else: 
            test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(test_indices, False))
            
        return data_loader

    def _setup_data(self, datasets, path, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []
        
        trsf_train = transforms.Compose(self.train_transforms)
        trsf_test = transforms.Compose(self.common_transforms)

        for dataset in datasets:
            if(self.dataset_name=="imagenet"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=False, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='val', download=False, transform=trsf_test)
                
            elif(self.dataset_name=="cub200" or self.dataset_name=="cifar100" or self.dataset_name=="mnist"  or self.dataset_name=="caltech101"  or self.dataset_name=="omniglot"  or self.dataset_name=="celeb"):
                train_dataset = dataset.base_dataset(root=path, train=True, download=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, train=False, download=True, transform=trsf_test)

            elif(self.dataset_name=="svhn"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='test', download=True, transform=trsf_test)
                train_dataset.targets = train_dataset.labels
                test_dataset.targets = test_dataset.labels
                
            order = [i for i in range(self.args.num_class)]
            if random_order:
                random.seed(seed)  
                random.shuffle(order)
            elif dataset.class_order is not None:
                order = dataset.class_order
                
            for i,t in enumerate(train_dataset.targets):
                train_dataset.targets[i] = order[t]
            for i,t in enumerate(test_dataset.targets):
                test_dataset.targets[i] = order[t]
            self.class_order.append(order)

            self.increments = [increment for _ in range(len(order) // increment)]

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def get_memory(self, memory, for_memory, seed=1):
        random.seed(seed)
        memory_per_task = self.args.memory // ((self.args.sess+1)*self.args.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1

        if(memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task*(self.args.sess)):
                idx = np.where(targets_memory==class_idx)[0][:memory_per_task]
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))   ])
                self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))    ])

        new_indices, new_targets = for_memory
        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task*(self.args.sess),self.args.class_per_task*(1+self.args.sess)):
            idx = np.where(new_targets==class_idx)[0][:memory_per_task]
            self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx],(mu,))   ])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx],(mu,))    ])
            
        print(len(self._data_memory))
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))
    
def _get_datasets():
    return [_get_dataset() ]

def _get_dataset():
    return iCIFAR100
class DataHandler:
    base_dataset = None
    train_transforms = []
    mata_transforms = [transforms.ToTensor()]
    common_transforms = [transforms.ToTensor()]
    class_order = None

class iCIFAR100(DataHandler):
    base_dataset = datasets.cifar.CIFAR100
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    


    

