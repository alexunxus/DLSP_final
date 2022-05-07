import os
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm

from src.model import WideResNet_2, WRN34_out_branch
from src.common import trim_dict
from src.dataset import CleanDataset, get_clean_test
from src.reverse_attack import reverse_pgd
from src.loss import compute_contrastive_loss

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.backends import cudnn
from torchvision import transforms

## accelerate computation
cudnn.benchmark = True
torch.manual_seed(0)
np.random.seed(0)

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--task', type=str, default="default", help="task type: [default|SSL|random]")
    argparser.add_argument("--norm", type=str, default="l_2", help='norm type: [clean|l_1|l_2|l_inf]')
    argparser.add_argument("--iter", type=int, default=5, help="number of SSL iteration: [5|10|15]")
    argparser.add_argument("--batchsize",   type=int, default=256)
    argparser.add_argument("--attack_iter", type=int, default=5)
    argparser.add_argument("--epsilon",     type=int, default=8)
    argparser.add_argument("--alpha",       type=int, default=2)
    args = argparser.parse_args()

    base_model = WideResNet_2(depth=28, widen_factor=10, single=True if args.task=='default' else False)
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
    if args.norm == 'clean':
        test_dataset = get_clean_test()
        test_loader  = DataLoader(test_dataset, batch_size= args.batchsize, shuffle=False, pin_memory=True, num_workers=4)
    else:
        data_base_dir = f"data/pgd/{args.norm}/"
        test_x = np.load(os.path.join(data_base_dir, f"test_perturbed_X_{args.norm}_{args.iter}.npy"))
        test_y = np.load(os.path.join(data_base_dir, f"test_perturbed_y_{args.norm}_{args.iter}.npy"))

        test_dataset = CleanDataset(X= test_x, y = test_y)
        test_loader  = DataLoader(test_dataset, batch_size= args.batchsize, shuffle=False, pin_memory=True, num_workers=4)
    
    # device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # criterion
    criterion = nn.CrossEntropyLoss().to(device)

    task = args.task
    if task not in ['default', 'SSL', 'random']:
        raise ValueError(f"Unknown task {task}")
    if task == 'default':
        # doing inference without self-supervised head
        base_model = base_model.to(device)
        base_model.eval()
        
        test_loss = 0
        test_acc  = 0
        counter   = 0

        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                pred = base_model(x)
                loss = criterion(pred, y)
                
                _, out    = torch.max(pred, dim=-1)
                test_acc  += (out == y).sum().item()
                test_loss += loss.item()*pred.shape[0]
                counter   += pred.shape[0]
            
        test_loss /= counter
        test_acc  /= counter
        
        with open(f"./weight/loss_{args.norm}.txt", 'w+') as f:
            f.write(f"{test_loss}:{test_acc}")

        print(f"Test[{args.norm}][{args.iter}] loss = {test_loss:.4f}, acc = {test_acc*100:.2f}")
    elif args.task == 'random':
        # doing inference without self-supervised head
        base_model = base_model.to(device)
        base_model.eval()
        
        test_loss = 0
        test_acc  = 0
        counter   = 0
        epsilon      = args.epsilon/255.
        alpha        = args.alpha/255.

        for x, y in test_loader:
            delta = torch.zeros_like(x)
            delta.uniform_(-epsilon, epsilon)
            x += delta

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                pred, hidden = base_model(x)
                loss = criterion(pred, y)
                
                _, out    = torch.max(pred, dim=-1)
                test_acc  += (out == y).sum().item()
                test_loss += loss.item()*pred.shape[0]
                counter   += pred.shape[0]
            
        test_loss /= counter
        test_acc  /= counter

        print(f"Test Random Reverse Attack: [{args.norm}][{args.iter}] loss = {test_loss:.4f}, acc = {test_acc*100:.2f}")
    else:
        # perform inference with self-supervision
        contrastive_head = WRN34_out_branch()
        
        # if load official weight
        head_state_path = "./weight/ssl_model_130.pth"
        state_info = trim_dict(torch.load(head_state_path)['ssl_model'])
        contrastive_head.load_state_dict(state_info)
        
        # if load self-trained weight
        #head_state_path = './weight/contrastive_head.h5'
        #state_info = torch.load(head_state_path)
        #contrastive_head.load_state_dict(state_info)
        contrastive_head = contrastive_head.to(device)
        base_model = base_model.to(device)
        
        contrastive_head.eval()
        base_model.eval()
        
        # script transform
        transform = torch.nn.Sequential(
            transforms.RandomResizedCrop(size=32),
            transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.2)
        )
        scripted_transforms = torch.jit.script(transform)
        
        # We use this to allow scripted transforms to be differentiable. Need this due to Pytorch Issue.
        for i, batch in enumerate(test_loader):
            x, _ = batch
            x = x.to(device)
            contrastive_Loss = \
                compute_contrastive_loss(x, base_model, contrastive_head, scripted_transforms, criterion)
            break

        # hyperparameter
        epsilon      = args.epsilon/255.
        alpha        = args.alpha/255.
        attack_iters = args.attack_iter

        test_loss = 0
        test_acc  = 0
        counter   = 0
        adv_loss  = []
        radv_loss = []

        i = 0
        tbar = tqdm(test_loader)
        for x, y in tbar:
            i += 1
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            r_delta, best_loss, init_loss = reverse_pgd(base_model, 
                                                        contrastive_head, 
                                                        scripted_transforms, 
                                                        criterion,
                                                        x,
                                                        epsilon, 
                                                        alpha, 
                                                        attack_iters, 
                                                        norm=args.norm, 
                                                        n_views=2)
            
            x = x + r_delta
            adv_loss.append(init_loss)
            radv_loss.append(best_loss)

            with torch.no_grad():
                pred, _ = base_model(x)
                loss = criterion(pred, y)
                
                _, out    = torch.max(pred, dim=-1)
                test_acc  += (out == y).sum().item()
                test_loss += loss.item()*pred.shape[0]
                counter   += pred.shape[0]
            
            torch.cuda.empty_cache()
            tbar.set_description(f"[{i}/{len(test_loader)}]test_acc = {test_acc/counter*100:.2f}%, test_loss = {test_loss/counter:.4f}")
            
        test_loss /= counter
        test_acc  /= counter

        np.save(f'./weight/loss_adv_{args.norm}_{args.iter}', adv_loss)
        np.save(f'./weight/loss_radv_{args.norm}_{args.iter}', radv_loss)

        with open(f"./weight/loss_{args.norm}_{args.iter}.txt", 'w+') as f:
            f.write(f"{test_loss}:{test_acc}")
            
        print(f"SSL Reverse Attack Test[{args.norm}][{args.iter}] loss = {test_loss:.4f}, acc = {test_acc*100:.2f}%")
