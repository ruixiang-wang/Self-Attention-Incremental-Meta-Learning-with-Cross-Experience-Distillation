from iCaRL import iCaRLmodel
from ResNet import resnet18_cbam
import torch

numclass=20
feature_extractor=resnet18_cbam()
img_size=32
batch_size=128
task_size=20
memory_size=2000
epochs=2
learning_rate=2.0

model=iCaRLmodel(numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate)
#model.model.load_state_dict(torch.load('model/ownTry_accuracy:84.000_KNN_accuracy:84.000_increment:10_net.pkl'))

for i in range(5):
    model.beforeTrain()
    accuracy=model.train()
    model.afterTrain(accuracy)
