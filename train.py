from argparse import ArgumentParser
import numpy as np
import os

from src.model import WideResNet, WideResNet_2, WRN34_out_branch
from src.pipeline import Trainer, ContrastiveTrainer
from src.dataset import split_dataset
from src.metrics import acc
from src.callbacks import CheckpointCallback, ContrastiveCheckpointCallback
from src.common import trim_dict
from src.loss import compute_contrastive_loss

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam, SGD
from torch.backends import cudnn
from torchvision import transforms
import torch



## accelerate computation
cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--task",      type=str,   default="SSL",  help="task type: [Clean|SSL], train clean classifier or self-supervised head")
    argparser.add_argument("--loss",      type=str,   default="Cosine", help="loss type: [Cosine|Dot]")
    argparser.add_argument("--batchsize", type=int,   default=1024)
    argparser.add_argument("--lr",        type=float, default=1e-4)
    argparser.add_argument("--epochs",    type=int,   default=200)
    argparser.add_argument("--optim",     type=str,   default="Adam")
    args = argparser.parse_args()
    
    task = args.task
    if task not in ['Clean', "SSL"]:
        raise ValueError(f"Unknown task {task}")
    if task == 'Clean':
        # 1. train clean classifier here
        # 2. save the model at ./weight/clean.h5
        
        # prepare model
        model = WideResNet(depth=16, num_classes=10)

        # prepare dataset
        cifar_train, train_sampler, valid_sampler = split_dataset()
        
        train_loader = DataLoader(cifar_train, batch_size=argparser.batchsize, sampler=train_sampler, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(cifar_train, batch_size=argparser.batchsize, sampler=valid_sampler, num_workers=4, pin_memory=True)

        # loss function
        criterion = nn.CrossEntropyLoss()

        # optimizer
        if argparser.optim == 'Adam':
            optim = Adam(model.parameters(), lr=args.lr, betas=[0.9, 0.99])
        elif argparser.optim == 'SGD':
            optim = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer {args.optim}")

        # callbacks
        callbacks = [CheckpointCallback(filepath='./weight/clean.h5', monitor='val_loss', save_best_only=True)]

        # scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [args.epochs//2, 3*args.epochs//4], gamma=0.1)

        # training
        trainer = Trainer(train_loader, model, criterion, optim, 10, args.epochs, [acc], scheduler = scheduler,
                          val_loader=valid_loader, log_path="/weight/clean.log", callbacks=callbacks)
        trainer.fit()



    else:
        # train self supervised model here
        # 1. build the model
        # 2. load the model weight from ./weight/clean.h5
        # 3. Load data from augmented loader

        base_model = WideResNet_2(depth=28, widen_factor=10)
        contrast_head = WRN34_out_branch()

        state_dict_path = "./weight/cifar10_rst_adv.pt.ckpt"
        if not os.path.isfile(state_dict_path):
            raise ValueError(f"Please download the model weight from https://cv.cs.columbia.edu/mcz/ICCVRevAttack/cifar10_rst_adv.pt.ckpt")
        loaded_info = torch.load(state_dict_path)
        n_classes   = loaded_info['num_classes']
        state_dict  = loaded_info['state_dict']
        # trim state dict here:
        state_dict = trim_dict(state_dict)
        base_model.load_state_dict(state_dict)
        

        # prepare dataset
        cifar_train, train_sampler, _ = split_dataset()
        
        train_loader = DataLoader(cifar_train, batch_size=args.batchsize, sampler=train_sampler, num_workers=4, pin_memory=True)
        
        transform = torch.nn.Sequential(
            transforms.RandomResizedCrop(size=32),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2)
        )
        scripted_transform = torch.jit.script(transform)

        # criterion
        criterion = nn.CrossEntropyLoss()

        # optimizer
        if args.optim == 'Adam':
            optim = Adam(contrast_head.parameters(), lr=args.lr, betas=[0.9, 0.99])
        elif args.optim == 'SGD':
            optim = SGD(contrast_head.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer {args.optim}")

        # scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [args.epochs//2, 3*args.epochs//4], gamma=0.1)

        # callbacks
        callbacks = [ContrastiveCheckpointCallback(filepath='./weight/contrastive_head.h5', monitor='loss', save_best_only=True)]

        trainer =  ContrastiveTrainer(train_loader       =train_loader, 
                                      model              = None, 
                                      base_model         = base_model,
                                      contrastive_head   = contrast_head,
                                      loss_fn            = compute_contrastive_loss,
                                      scripted_transform = scripted_transform,
                                      criterion          = criterion, 
                                      optim              = optim, 
                                      nclass             = n_classes, 
                                      epochs             = args.epochs, 
                                      metric_fns         = [], 
                                      scheduler          = scheduler,
                                      log_path           = "/weight/train_ssl.log", 
                                      callbacks          = callbacks)
        
        trainer.fit()
        
