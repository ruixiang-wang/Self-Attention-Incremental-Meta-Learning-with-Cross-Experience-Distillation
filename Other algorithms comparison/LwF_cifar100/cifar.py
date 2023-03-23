# 导入库
import pickle
# pickle库是标准库，他的主要作用是处理复杂的序列化语法，序列化时，只是序列化整个序列对象，而非内存地址。
#之后我用的pickle.load，这个的作用是读取文件。读取时，pickle.load返回的是一个字典，file.read返回的是一个字符串。
import numpy as np
#numpy是一个非常实用的数学函数库
import os
#os是一个非常使用的文件操作库

# 将CIAFR-100的数据集进行处理，将这个数据集处理成5大块，每一块有20类
# 以进行增量学习
class Cifar100:
    def __init__(self):
        # 打开数据集并进行处理
        with open('cifar100/train','rb') as f:  #以二进制阅读的权限打开训练集为f
            self.train = pickle.load(f, encoding='latin1') #定义train属性，这是一个字典
        with open('cifar100/test','rb') as f:  #以二进制阅读的权限打开测试集为f
            self.test = pickle.load(f, encoding='latin1')

        # 获取到她们的数据和标签
        self.train_data = self.train['data']
        self.train_labels = self.train['fine_labels']
        self.test_data = self.test['data']
        self.test_labels = self.test['fine_labels']

        # 开始定义训练数据集以及测试数据集、以及每一次训练的数据集
        self.train_groups, self.each_train_groups, self.test_groups = self.initialize()
        # 增量次数
        self.batch_num = 5

    def initialize(self):
        # 用来存储所有的数据
        train_groups = [[],[],[],[],[]] #因为增量次数是五次，所以分为五类
        # 正常情况下，CIFAR100有100类、我为了测试每一类的准确率
        # 如果是cifar10的话，就应该改为10类
        each_train_groups = [[] for _ in range(100)]

        # 我要按照对应的顺序处理混乱的数据集 zip的作用是将其打包成元组，互相对应
        for train_data, train_label in zip(self.train_data, self.train_labels):
            # print(train_data.shape)

            # rgb处理、处理图片
            train_data_r = train_data[:1024].reshape(32, 32)
            train_data_g = train_data[1024:2048].reshape(32, 32)
            train_data_b = train_data[2048:].reshape(32, 32)

            # 合并图片，数据预处理 dstack的作用是将列表中的数组沿深度方向进行拼接。
            train_data = np.dstack((train_data_r, train_data_g, train_data_b))

            # 分类按类别把数据进行处理
            # 如果标签小于20，那么我将他们放在[0]中
            if train_label < 20:
                train_groups[0].append((train_data,train_label))
                each_train_groups[train_label].append((train_data,train_label))
            elif 20 <= train_label < 40:
                train_groups[1].append((train_data,train_label))
                each_train_groups[train_label].append((train_data,train_label))
            elif 40 <= train_label < 60:
                train_groups[2].append((train_data,train_label))
                each_train_groups[train_label].append((train_data,train_label))
            elif 60 <= train_label < 80:
                train_groups[3].append((train_data,train_label))
                each_train_groups[train_label].append((train_data,train_label))
            elif 80 <= train_label < 100:
                train_groups[4].append((train_data,train_label))
                each_train_groups[train_label].append((train_data,train_label))

        # 进行判断，以保证数据是预处理正确的
        assert len(train_groups[0]) == 10000, len(train_groups[0])
        assert len(train_groups[1]) == 10000, len(train_groups[1])
        assert len(train_groups[2]) == 10000, len(train_groups[2])
        assert len(train_groups[3]) == 10000, len(train_groups[3])
        assert len(train_groups[4]) == 10000, len(train_groups[4])
        assert len(each_train_groups[40]) == 500, len(each_train_groups[40])

        # 处理测试集的数据
        test_groups = [[],[],[],[],[]]
        for test_data, test_label in zip(self.test_data, self.test_labels):
            test_data_r = test_data[:1024].reshape(32, 32)
            test_data_g = test_data[1024:2048].reshape(32, 32)
            test_data_b = test_data[2048:].reshape(32, 32)
            test_data = np.dstack((test_data_r, test_data_g, test_data_b))
            if test_label < 20:
                test_groups[0].append((test_data,test_label))
            elif 20 <= test_label < 40:
                test_groups[1].append((test_data,test_label))
            elif 40 <= test_label < 60:
                test_groups[2].append((test_data,test_label))
            elif 60 <= test_label < 80:
                test_groups[3].append((test_data,test_label))
            elif 80 <= test_label < 100:
                test_groups[4].append((test_data,test_label))
        assert len(test_groups[0]) == 2000
        assert len(test_groups[1]) == 2000
        assert len(test_groups[2]) == 2000
        assert len(test_groups[3]) == 2000
        assert len(test_groups[4]) == 2000

        return train_groups, each_train_groups, test_groups
        # 对需要的进行返回

    # 增量学习中每一次只能新增一堆数据，可以控制每一组数据
    def getNextClasses(self, i):
        return self.train_groups[i], self.test_groups[i]

    def eachclass(self, i):
        return self.each_train_groups[i]

if __name__ == "__main__":
    cifar = Cifar100()
    #print(len(cifar.each_train_groups[30]))
