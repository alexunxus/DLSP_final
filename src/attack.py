import os
from argparse import ArgumentParser
import numpy as np
from model import WideResNet_2,WRN34_out_branch
import numpy as np
import torch
from torch import nn
import torchvision

from common import trim_dict, clamp, Batches

import torch
import torch.nn.functional as F


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False):
    upper_limit, lower_limit = 1, 0
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        elif norm == "l_1":
            pass
        else:
            raise ValueError
        delta = clamp(delta, lower_limit - X, upper_limit - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output, _ = model(X+delta)
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = criterion(output, y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            elif norm == "l_1":
                g_norm = torch.sum(torch.abs(g.view(g.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)

            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()

        all_loss = F.cross_entropy(model(X+delta)[0], y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta, X+max_delta


def attack_pgd_main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_set = torchvision.datasets.CIFAR10(root="./data/cifar10/", 
                                            train=False, download=True)
    
    cifar_test = {'data': test_set.data, 'labels': test_set.targets}
    cifar_test['data'] = torch.Tensor(cifar_test['data'])
    cifar_test['data'] = torch.moveaxis(cifar_test['data'],-1,1)
    
    test_set = list(zip(cifar_test['data'] / 255., cifar_test['labels']))
    
    test_batches = Batches(test_set, 64, shuffle=False, num_workers=2)
    base_model = WideResNet_2(depth=28, widen_factor=10)
    state_dict_path = "./weight/cifar10_rst_adv.pt.ckpt"
    if not os.path.isfile(state_dict_path):
        raise ValueError(
            f"Please download the model weight from https://cv.cs.columbia.edu/mcz/ICCVRevAttack/cifar10_rst_adv.pt.ckpt")
    loaded_info = torch.load(state_dict_path)
    state_dict = loaded_info['state_dict']
    # trim state dict here:
    state_dict = trim_dict(state_dict)
    base_model.load_state_dict(state_dict)
    base_model = base_model.to(device)
    
    all_delta = []
    testX = []
    testy = []
    new_test = []
    for i, batch in enumerate(test_batches):
        X,y = batch['input'],batch['target']
        testX.append(X)
        testy.append(y)
        delta,new = attack_pgd(base_model, X, y, args.epsilon, args.pgd_alpha, args.attack_iters, args.restarts, args.norm,
                           early_stop=args.eval)
        all_delta.append(delta)
        new_test.append(new)
        print("Done with :", (i+1)*64)
    testX = torch.cat(testX, dim = 0)
    testy = torch.cat(testy, dim = 0)
    all_delta = torch.cat(all_delta, dim = 0)
    new_test = torch.cat(new_test, dim = 0)

    base_path = os.path.join(args.savepath, f"{args.norm}/")
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    np.save(os.path.join(base_path, f"Test_perturbed_X_{args.norm}_{args.attack_iters}.npy"), new_test.to('cpu')) 
    np.save(os.path.join(base_path, f"Test_perturbed_y_{args.norm}_{args.attack_iters}.npy"), testy.to('cpu')) 

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--epsilon', default=8, type=int)
    argparser.add_argument('--attack-iters', default=10, type=int)
    argparser.add_argument('--pgd-alpha', default=2, type=float)
    argparser.add_argument('--l2', default=0, type=float)
    argparser.add_argument('--l1', default=0, type=float)
    argparser.add_argument('--batch-size', default=1024, type=int)
    argparser.add_argument('--restarts', default=1, type=int)
    argparser.add_argument('--norm', default='l_2', type=str, choices=['l_inf', 'l_2', 'l_1'])
    argparser.add_argument('--eval', action='store_true')
    argparser.add_argument('--savepath', type=str, default="./data/")
    args = argparser.parse_args()
    print(args.epsilon/255., args.pgd_alpha/255., args.attack_iters, args.restarts, args.norm,args.eval)
    attack_pgd_main(args)