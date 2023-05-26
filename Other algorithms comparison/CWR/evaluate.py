import sys
import time
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from MobileNetV2 import MobileNetV2

# load mobilenet without the last fc layer
net = MobileNetV2(n_class=51)
if torch.cuda.is_available():
    net = net.cuda()
    loaded_dict = torch.load('mobilenet_v2.pth.tar')  # add map_location='cpu' if no gpu
else:
    loaded_dict = torch.load('mobilenet_v2.pth.tar', map_location='cpu')
state_dict = {k: v for k, v in loaded_dict.items() if k in net.state_dict()}
state_dict["classifier.1.weight"] = net.state_dict()["classifier.1.weight"]
state_dict["classifier.1.bias"] = net.state_dict()["classifier.1.bias"]
net.load_state_dict(state_dict)

# determine optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# check if cuda exist then use cuda otherwise use cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


# Construct pytorch dataset class
class BatchData(Dataset):

    def format_images(self, path, datatype, batch_index):
        path_prefix = '{}/{}/batch{}/'.format(path, datatype, batch_index)
        path_prefix = os.path.join(path,datatype,'batch'+str(batch_index))+'/'
        table = pd.read_csv(path_prefix + 'label.csv', index_col=0)
        data_list = [path_prefix + filename for filename in table['file name'].tolist()]
        label_list = table['label'].tolist()

        return data_list, label_list

    def __init__(self, path, datatype, batch_index, transforms):
        self.transforms = transforms
        self.data_list, self.label_list = self.format_images(path, datatype, batch_index)

        # print a summary
        print('Load {} batch {} have {} images '.format(datatype, batch_index, len(self.data_list)))

    def __getitem__(self, idx):
        img = self.data_list[idx]
        img = Image.open(img)
        label = int(self.label_list[idx])
        img = self.transforms(img)
        return img, label, self.data_list[idx].split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.data_list)


def feed(dataloader, is_training, num_epochs=50, validloader=None, is_valid=False, is_save_csv=False, csv_name=None):
    losses = list()
    acces = list()
    filename = list()
    label_gt = list()
    label_predict = list()

    since = time.time()
    start = time.time()
    if is_training:
        net.train()
    else:
        num_epochs = 1
        net.eval()

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        train_acc = 0.0
        i = 0
        for data in dataloader:
            i += 1
            # get the inputs
            inputs, labels, file = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)

            if is_training:
                optimizer.step()

            _, pred = outputs.max(1)

            num_correct = (pred == labels).sum().item()
            acc = num_correct / inputs.shape[0]
            # For csv output
            label_predict.extend(pred.tolist())
            label_gt.extend(labels.tolist())
            filename.extend(list(file))

        acces.append(train_acc / len(dataloader))

        time_elapsed = time.time() - since
        since = time.time()

        if is_training:
            valid_acc, valid_loss = valid(validloader)
            print(
                'epoch{}/{} time:{:.0f}m {:.0f}s train_acc:{:.3f} train_loss:{:.4f} valid_acc:{:.3f} valid loss:{:.3f}'.format(
                    epoch + 1,
                    num_epochs, time_elapsed // 60,
                    time_elapsed % 60,
                    train_acc / len(dataloader),
                    running_loss / len(dataloader),
                    valid_acc,
                    valid_loss))
        elif not is_valid:
            print('epoch{}/{} time:{:.0f}m {:.0f}s acc:{:.3f} loss:{:.4f}'.format(epoch + 1,
                                                                                  num_epochs, time_elapsed // 60,
                                                                                  time_elapsed % 60,
                                                                                  train_acc / len(dataloader),
                                                                                  running_loss / len(dataloader)))

        if is_save_csv and csv_name != None:
            column = {'file': filename,
                      'label_predict': label_predict}
            log = pd.DataFrame(column)
            print('write test csv to {}'.format(csv_name))
            log.to_csv(csv_name)
    if is_training:
        time_elapsed = time.time() - start
        print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_acc / len(dataloader), running_loss / len(dataloader)


def valid(validloader):
    return feed(validloader, is_training=False, is_valid=True)


def test(testloader, is_save_csv=False, csv_name=None):
    feed(testloader, is_training=False, is_save_csv=is_save_csv, csv_name=csv_name)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('[usage] python evaluate.py dataset_path output_path')
        exit(0)
    dataset_path = sys.argv[1]
    output_path = sys.argv[2]
    print(dataset_path, output_path)

    # Image preprocessing
    trans = transforms.Compose([
        #                     transforms.Resize((300,300)),
        transforms.RandomSizedCrop(255),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_batch_list = [BatchData(dataset_path, 'test', i, trans) for i in range(1, 10)]

    test_loader_list = [torch.utils.data.DataLoader(batch, batch_size=16, shuffle=False, num_workers=2)
                        for batch in test_batch_list]

# load your models
    state_dict = torch.load('./model/model_1.pth', map_location='cpu')
    net.load_state_dict(state_dict)

    for test_task in range(9):
        print('[Test in task{}]:'.format(test_task + 1))
        test(test_loader_list[test_task], is_save_csv=True,
             csv_name=os.path.join(output_path,'test_batch'+str(test_task+1)+'.csv'))
