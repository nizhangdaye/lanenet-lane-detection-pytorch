import time
import os
import sys

import torch
from model.lanenet.train_lanenet import train_model
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from torchvision import transforms

from model.utils.cli_helper import parse_args
from model.eval_function import Eval_Score

import numpy as np
import pandas as pd
import cv2

import torch.nn.functional as F

# 对 python 多进程的一个 pytorch 包装
import torch.multiprocessing as mp

# 这个 sampler 可以把采样的数据分散到各个 CPU 上
from torch.utils.data.distributed import DistributedSampler

# 实现分布式数据并行的核心类
from torch.nn.parallel import DistributedDataParallel as DDP

# DDP 在每个 GPU 上运行一个进程，其中都有一套完全相同的 Trainer 副本（包括model和optimizer）
# 各个进程之间通过一个进程池进行通信，这两个方法来初始化和销毁进程池
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup():
    # torchrun 会处理环境变量以及 rank & world_size 设置
    os.environ["MASTER_ADDR"] = "localhost"  # 由于这里是单机实验所以直接写 localhost
    # os.environ["MASTER_PORT"] = "12355"  # 任意空闲端口
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def train():
    gpu_id = int(os.environ['LOCAL_RANK'])

    args = parse_args()
    save_path = args.save
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    resize_height = args.height
    resize_width = args.width

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((resize_height, resize_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    target_transforms = transforms.Compose([
        Rescale((resize_width, resize_height)),
    ])

    train_dataset = TusimpleSet(train_dataset_file, transform=data_transforms['train'],
                                target_transform=target_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False, drop_last=True,
                              sampler=DistributedSampler(train_dataset))

    val_dataset = TusimpleSet(val_dataset_file, transform=data_transforms['val'],
                              target_transform=target_transforms)
    val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False, drop_last=True,
                            sampler=DistributedSampler(val_dataset))

    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    dataset_sizes = {'train': len(train_loader.dataset), 'val': len(val_loader.dataset)}

    model = LaneNet(arch=args.model_type)
    model.to(gpu_id)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")

    model = DDP(model, device_ids=[gpu_id])

    model, log = train_model(model, optimizer, scheduler=None, dataloaders=dataloaders, dataset_sizes=dataset_sizes,
                             device=gpu_id, loss_type=args.loss_type, num_epochs=args.epochs)

    df = pd.DataFrame({'epoch': [], 'training_loss': [], 'val_loss': []})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']

    train_log_save_filename = os.path.join(save_path, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch', 'training_loss', 'val_loss'], header=True, index=False,
              encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))

    model_save_filename = os.path.join(save_path, 'best_model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))


def main():
    # 初始化进程池
    ddp_setup()

    train()

    # 销毁进程池
    destroy_process_group()


if __name__ == '__main__':
    main()
