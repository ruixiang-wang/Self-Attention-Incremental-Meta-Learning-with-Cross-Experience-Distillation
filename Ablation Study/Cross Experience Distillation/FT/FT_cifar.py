import argparse
import os.path as osp
import os

import pickle as pkl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from models.protonet import ProtoNet
from torch.utils.data import DataLoader
from utils import (
    pprint,
    set_gpu,
    ensure_path,
    make_path,
    Averager,
    Timer,
    count_acc,
    compute_confidence_interval,
    save_model,
)


def train(epoch, model, train_loader, optimizer):
    model.train()
    tl = Averager()
    ta = Averager()

    label = torch.arange(args.way).repeat(args.query)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)

    for i, batch in enumerate(train_loader, 1):
        if torch.cuda.is_available():
            data, _ = [_.cuda() for _ in batch]
        else:
            data = batch[0]

        p = args.shot * args.way
        data_shot, data_query = data[:p], data[p:]
        logits = model(data_shot, data_query)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        if i % 50 == 0:
            print(
                "epoch {}, train {}/{}, loss={:.4f} acc={:.4f}".format(
                    epoch, i, len(train_loader), loss.item(), acc
                )
            )

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return tl,ta



def main():
    THIS_PATH = osp.dirname(__file__)
    if args.dataset=='CIFAR100':
        ROOT_PATH = osp.join(THIS_PATH, "data/cifar")
        TOTAL_TASK_NUM = 100
        META_TEST_CLS_NUM = 20

    TASK_NUM=args.TASK_NUM
    BASE_CLS_NUM=args.BASE_CLS_NUM

    if TASK_NUM > 1:
        CLS_NUM_PER_TASK = (TOTAL_TASK_NUM - BASE_CLS_NUM - META_TEST_CLS_NUM) // (TASK_NUM - 1)
    else:
        BASE_CLS_NUM = TOTAL_TASK_NUM-META_TEST_CLS_NUM

    file_name = '_'.join(('base', str(BASE_CLS_NUM),
                          'task', str(TASK_NUM), 'meta_test',
                          str(META_TEST_CLS_NUM)))+'.pkl'

    with open(osp.join(ROOT_PATH, file_name),'rb') as h:
        all_data=pkl.load(h)

    model = ProtoNet(args)

    for session in range(0,TASK_NUM):
        if session>=1:
            ckpt_name = '_'.join(('session',str(session-1),'base', str(BASE_CLS_NUM), 'task', str(TASK_NUM),
                                'meta_test', str(META_TEST_CLS_NUM), 'method', str(args.method)))

            state_dict = torch.load(osp.join(args.save_path, ckpt_name + ".pth"), map_location='cpu')
            model.load_state_dict(state_dict['params'])

        curr_task_data=all_data[session]
        train_label=curr_task_data['train_label']
        train_data=curr_task_data['train_data']

        if session==0:
            batch_size=args.batch_size*args.batch_times
        else:
            batch_size=args.batch_size

        trainset = Dataset("train", args, train_data,train_label,location=args.location)
        train_sampler = CategoriesSampler(
            trainset.label, batch_size, args.way, args.shot + args.query,cls_per_task=args.cls_per_task
        )
        train_loader = DataLoader(
            dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.lr_decay:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=args.step_size, gamma=args.gamma
            )

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            model = model.cuda()

        timer = Timer()

        for epoch in range(1, args.max_epoch + 1):

            tl,ta=train(epoch, model, train_loader, optimizer)
            if epoch % 10 == 0:
                print(
                    "ETA:{}/{}".format(timer.measure(), timer.measure(epoch / args.max_epoch))
                )

            if args.lr_decay:
                lr_scheduler.step()

        ckpt_name = '_'.join(('session',str(session),'base', str(BASE_CLS_NUM), 'task', str(TASK_NUM),
                              'meta_test', str(META_TEST_CLS_NUM),'method', str(args.method)))
        save_model(model,ckpt_name,args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument(
        "--model",
        type=str,
        default="convnet",
        choices=[
            "convnet",
            "resnet12",
            "resnet18",
            "resnet34",
            "densenet121",
            "wideres",
        ],
    )
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=15)
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--validation_way", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument(
        "--dataset", type=str, default="CIFAR100",
        choices=["MiniImageNet", "CUB", "CIFAR100"]
    )
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--hyperbolic", action="store_true", default=False)
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=1600)
    parser.add_argument("--location", type=int, default=105)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--batch_times", type=int, default=1)

    ##############################################
    parser.add_argument("--init_weights", type=str, default=None)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--lr_decay", type=bool, default=True)
    parser.add_argument("--train_c", action="store_true", default=False)
    parser.add_argument("--train_x", action="store_true", default=False)
    parser.add_argument("--not_riemannian", action="store_true")

    parser.add_argument("--TASK_NUM", type=int, default=16)
    parser.add_argument("--cls_per_task", type=int, default=5)
    parser.add_argument("--BASE_CLS_NUM", type=int, default=5)
    parser.add_argument("--method", type=str, default='intra')
    ################## for cosine linear ###################
    parser.add_argument("--norm", action="store_true")
    parser.add_argument("--sigma", action="store_true")
    parser.add_argument("--reverse", action="store_true")

    args = parser.parse_args()
    pprint(vars(args))
    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")

    if not os.path.exists('logs'):
        os.makedirs('logs')

    if args.save_path is None:
        save_path1 = "logs/"+"-".join([args.dataset, "ProtoNet"])
        save_path2 = "_".join(
            [
                str(args.shot),
                str(args.query),
                str(args.way),
                str(args.validation_way),
                str(args.step_size),
                str(args.gamma),
                str(args.lr),
                str(args.temperature),
                str(args.reverse),
                str(args.dim),
                str(args.c)[:5],
                str(args.train_c),
                str(args.train_x),
                str(args.model),
                str(args.TASK_NUM),
                str(args.method),
            ]
        )
        args.save_path = save_path1 + "_" + save_path2
        make_path(args.save_path)
    else:
        make_path(args.save_path)

    if args.dataset=='CIFAR100':
        from dataloader.my_cifar import CIFAR100 as Dataset
    else:
        raise ValueError("Non-supported Dataset.")

    from dataloader.samplers import CategoriesSampler
    main()