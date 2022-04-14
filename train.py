from argparse import ArgumentParser
from email.policy import default
from gc import callbacks

from src.model import WideResNet
from src.pipeline import Trainer
from src.dataset import CleanDataset
from src.loss import cosine_sim_loss
from src.metrics import acc
from src.callbacks import CheckpointCallback

from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam, SGD

argparser = ArgumentParser()
argparser.add_argument("--task", type=str, default="Clean",  help="task type: [Clean|SS], train clean classifier or self-supervised head")
argparser.add_argument("--loss", type=str, default="Cosine", help="loss type: [Cosine|Dot]")
argparser.add_argument("--batchsize", type=int, default=64)
argparser.add_argument("--lr", type=float, default=3e-4)
argparser.add_argument("--optim", type=str, default="Adam")


if __name__ == "__main__":
    task = argparser.task
    if task not in ['Clean', "SS"]:
        raise ValueError(f"Unknown task {task}")
    if task == 'Clean':
        # 1. train clean classifier here
        # 2. save the model at ./weight/clean.h5
        model = WideResNet(depth=16, num_classes=10)

        train_dataset = CleanDataset("train")
        valid_dataset = CleanDataset("valid")

        train_loader = DataLoader(train_dataset, batch_size=argparser.batchsize, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=argparser.batchsize, shuffle=False, num_workers=4, pin_memory=True)

        criterion = nn.CrossEntropyLoss()

        if argparser.optim == 'Adam':
            optim = Adam(model.parameters(), lr=argparser.lr, betas=[0.9, 0.99])
        elif argparser.optim == 'SGD':
            optim = SGD(model.parameters(), lr=argparser.lr, momentum=0.9, weight_decay=0.01)

        callbacks = [CheckpointCallback(filepath='./weight/clean.h5', monitor='val_loss', save_best_only=True)]
        
        trainer = Trainer(model, model, criterion, optim, 10, 30, [acc], 
                          val_loader=valid_loader, log_path="/weight/clean.log", callbacks=callbacks)
        trainer.fit()



    else:
        # train self supervised model here
        # 1. build the model
        # 2. load the model weight from ./weight/clean.h5
        # 3. Load data from augmented loader
        pass
