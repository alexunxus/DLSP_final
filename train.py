from argparse import ArgumentParser
import numpy as np

from src.model import WideResNet
from src.pipeline import Trainer
from src.dataset import split_dataset
from src.loss import constrastive_loss_func
from src.metrics import acc
from src.callbacks import CheckpointCallback

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam, SGD
from torch.backends import cudnn
import torch

argparser = ArgumentParser()
argparser.add_argument("--task",      type=str,   default="Clean",  help="task type: [Clean|SS], train clean classifier or self-supervised head")
argparser.add_argument("--loss",      type=str,   default="Cosine", help="loss type: [Cosine|Dot]")
argparser.add_argument("--batchsize", type=int,   default=64)
argparser.add_argument("--lr",        type=float, default=1e-4)
argparser.add_argument("--optim",     type=str,   default="Adam")


## accelerate computation
cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

if __name__ == "__main__":
    task = argparser.task
    if task not in ['Clean', "SS"]:
        raise ValueError(f"Unknown task {task}")
    if task == 'Clean':
        # 1. train clean classifier here
        # 2. save the model at ./weight/clean.h5
        
        # prepare model
        model = WideResNet(depth=16, num_classes=10)

        # prepare dataset
        cifar_train, train_sampler, valid_sampler = split_dataset()
        
        train_loader = DataLoader(cifar_train, batch_size=argparser.batchsize, sampler=train_sampler, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(cifar_train, batch_size=argparser.batchsize, sampler=valid_sampler, shuffle=False, num_workers=4, pin_memory=True)

        # loss function
        criterion = nn.CrossEntropyLoss()

        # optimizer
        if argparser.optim == 'Adam':
            optim = Adam(model.parameters(), lr=argparser.lr, betas=[0.9, 0.99])
        elif argparser.optim == 'SGD':
            optim = SGD(model.parameters(), lr=argparser.lr, momentum=0.9, weight_decay=0.01)
        else:
            raise ValueError(f"Unknown optimizer {argparser.optim}")

        # scheduler
        

        # callbacks
        callbacks = [CheckpointCallback(filepath='./weight/clean.h5', monitor='val_loss', save_best_only=True)]
        
        # training
        trainer = Trainer(model, model, criterion, optim, 10, 30, [acc], 
                          val_loader=valid_loader, log_path="/weight/clean.log", callbacks=callbacks)
        trainer.fit()



    else:
        # train self supervised model here
        # 1. build the model
        # 2. load the model weight from ./weight/clean.h5
        # 3. Load data from augmented loader
        pass
