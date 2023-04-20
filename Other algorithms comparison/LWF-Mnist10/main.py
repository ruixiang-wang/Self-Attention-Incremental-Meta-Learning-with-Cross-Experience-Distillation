import torch
import numpy as np
from trainer import Trainer
import sys
from utils import *
import argparse

# str.encode('utf-8')
# bytes.decode('utf-8')

parser = argparse.ArgumentParser(description='Incremental Learning BIC')
parser.add_argument('--batch_size', default = 128, type = int)
parser.add_argument('--epoch', default = 255 , type = int)
parser.add_argument('--lr', default = 0.1, type = int)
parser.add_argument('--max_size', default = 2000, type = int)
parser.add_argument('--total_cls', default = 100, type = int)
args = parser.parse_args()


if __name__ == "__main__":
    # showGod()
    trainer = Trainer(args.total_cls)
    trainer.train(args.batch_size, args.epoch, args.lr, args.max_size)
