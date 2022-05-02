from email.mime import base
import os
from argparse import ArgumentParser
import numpy as np

from src.model import WideResNet_2
from src.common import trim_dict
from src.dataset import CleanDataset

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--task', type=str, help="task type: [default|SSL]")
    argparser.add_argument("--norm", type=str, default="l2", help='norm type: [l_1|l_2|l_inf]')
    argparser.add_argument("--iter", type=int, default=5, help="number of SSL iteration: [5|10|15]")
    argparser.add_argument("--batchsize", type=int, default=512)
    args = argparser.parse_args()

    base_model = WideResNet_2(depth=28, widen_factor=10)
    state_dict_path = "./weight/cifar10_rst_adv.pt.ckpt"
    
    if not os.path.isfile(state_dict_path):
        raise ValueError(f"Please download the model weight from https://cv.cs.columbia.edu/mcz/ICCVRevAttack/cifar10_rst_adv.pt.ckpt")
    loaded_info = torch.load(state_dict_path)
    n_classes   = loaded_info['num_classes']
    state_dict  = loaded_info['state_dict']

    # trim state dict here:
    state_dict = trim_dict(state_dict)
    base_model.load_state_dict(state_dict)

    # prepare test data here
    data_base_dir = f"data/Perturbed_data/{args.norm}/"
    test_x = np.load(os.path.join(data_base_dir, f"Test_perturbed_X_{args.norm}_{args.iter}.npy"))
    test_y = np.load(os.path.join(data_base_dir, f"Test_perturbed_y_{args.norm}_{args.iter}.npy"))

    test_dataset = CleanDataset(X= test_x, y = test_y)
    test_loader  = DataLoader(test_dataset, batch_size= args.batchsize, shuffle=False, pin_memory=True, num_workers=4)

    criterion = nn.CrossEntropyLoss()

    task = args.task
    if task not in ['default', 'SSL']:
        raise ValueError(f"Unknown task {task}")
    if task == 'default':
        # doing inference without self-supervised head
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        base_model = base_model.to(device)
        
        test_loss = 0
        test_acc  = 0
        counter   = 0

        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                pred = base_model(x)
                loss = criterion(pred, y)

                _, out = torch.max(pred)
                test_acc += (out == y).item()
                test_loss += loss.item()
                counter += pred.shape[0]
            
        test_loss /= counter
        test_acc  /= counter

        print(f"Test[{args.norm}][{args.iter}] loss = {test_loss:.4f}, acc = {test_acc*100:.2f}")
    else:
        # perform inference with self-supervision
        pass
