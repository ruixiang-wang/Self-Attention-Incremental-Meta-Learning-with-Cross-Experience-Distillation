import torch
import numpy as np
from trainer import Trainer
import sys #python标准库，作用是参数的交互
from utils import *
import argparse #python标准库，用于命令行解析模块
# 导入库

# 定义相关的参数，并且能作用于全局
parser = argparse.ArgumentParser(description='Incremental Learning LWF')
# 批大小，一次训练模型训练128张
parser.add_argument('--batch_size', default = 128, type = int) 
# 训练多少次
parser.add_argument('--epoch', default = 255 , type = int)
# 学习速率0.1
parser.add_argument('--lr', default = 0.1, type = int)
# 保存旧数据集的时候用到过
parser.add_argument('--max_size', default = 2000, type = int)
# 总共类别有多少
parser.add_argument('--total_cls', default = 100, type = int)
# 对上面的内容进行整合
args = parser.parse_args()


if __name__ == "__main__":
    # showGod()
    trainer = Trainer(args.total_cls)
    trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
