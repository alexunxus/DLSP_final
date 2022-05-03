import torch
from loss import compute_contrastive_loss

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def reverse_pgd(base_model, contrastive_head, scripted_transforms, criterion, 
                X, epsilon, alpha, attack_iters, norm='l_2', n_views=2):
    """
    Reverse algorithm that optimize the SSL loss via PGD
        base_model          : backbone model
        contrastive_head    : contrastive head
        scripted_transforms : augmentations
        criterion           : Crossentropy loss
        X                   : input data
        epsilon             : perturbation threshold
        alpha               : attack strength
        attack_iters        : number of iterations
        norm                : string: l_2, l_1, l_inf
        n_views             : number of views         
    """

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    delta  = torch.zeros_like(X).to(device)

    if norm == "l_inf":
        delta.uniform_(-epsilon, epsilon)
    elif norm == "l_2":
        delta.normal_()
        d_flat = delta.view(delta.size(0),-1)
        n      = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
        r      = torch.zeros_like(n).uniform_(0, 1)
        delta  *= r/n*epsilon
    elif norm == 'l_1':
        pass
    else:
        raise ValueError(f"Unknown normalization type {norm}")

    delta = clamp(delta, lower_limit-X, upper_limit-X)
    delta.requires_grad = True

    for _ in range(attack_iters):

        new_x = X + delta
        
        # TODO: here the neg sample is fixed, we can also try random neg sample to enlarge and diversify
        loss = -compute_contrastive_loss(new_x, 
                                         base_model, 
                                         contrastive_head, 
                                         scripted_transforms, 
                                         criterion, 
                                         n_views = n_views, 
                                         no_grad = False )

        loss.backward()
        grad = delta.grad.detach()

        d = delta
        g = grad
        x = X
        if norm == "l_inf":
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
        elif norm == "l_2":
            g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
            scaled_g = g/(g_norm + 1e-10)
            d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
        elif norm == "l_1":
            g_norm = torch.sum(torch.abs(g.view(g.shape[0], -1)), dim=1).view(-1, 1, 1, 1)
            scaled_g = g / (g_norm + 1e-10)
            d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=1, dim=0, maxnorm=epsilon).view_as(d)

        d = clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
        delta.grad.zero_()
    max_delta = delta.detach()
    
    return max_delta