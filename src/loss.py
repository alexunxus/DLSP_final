from matplotlib.pyplot import axis
import torch
import torch.functional as F


def normal_guassian_normalize(T):
    return (T-T.mean()) / T.std()

def clamp(X, lower_limit = 0, upper_limit = 1):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def contrastive_loss_func(contrastive_head, criterion, batchsize, n_views, temperature=0.2):
    """ 
        Loss function for contrastive SSL learning
        contrastive_head: (B*n_views, feature_size) Tensor
        criterion:        can said to be BCELoss
        batchsize:        int
        n_view:           number of augmentation perimage
        temperature:      tau for contrastive loss
    """
    # if don't want cosine similarity but dot product,
    # then don't use l2 normalize here
    features = F.normalize(contrastive_head, dim=1)

    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # labels.shape = (n_views * bathsize) -> (n_views * batch_size)
    # labels: [1, 2, 3, ...., batchsize-1]
    # labels.unsqueeze(0): [[1, 2, 3, ...., batchsize-1], 
    #                       [1, 2, 3, ...., batchsize-1], ...]
    # labels.unsqueeze(1): [[1, 1, 1, 1, ........], 
    #                       [2, 2, 2, 2, ........],]
    labels = torch.cat([torch.arange(batchsize) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    # similarity_matrix: shape (batch_size * batch_size)
    similarity_matrix = torch.matmul(features, features.T)

    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    logits = logits / temperature

    xcontrast_loss = criterion(logits, labels)

    correct = (logits.max(1)[1] == labels).sum().item()
    return xcontrast_loss, correct


def compute_contrastive_loss(x, base_model, contrastive_head, scripted_transforms, criterion,
                             n_views = 4, no_grad = True):
    '''
    Args:
        x:                 (B, d)-shaped tensor
        base_model:        take x and perform feature extraction
        contrastive_head:  take basemodel(x) and perform dimension reduction
        n_views:           number of repeats in x
        criterion:         BCELoss
        scripted_transforms: augmentations
    '''
    batch_size = x.shape[0]
    x = torch.cat([ scripted_transforms(x) for _ in range(n_views)], axis=0)

    if no_grad:
        with torch.no_grad():
            _, out = base_model(x)
    else:
        _, out = base_model(x)

    out = contrastive_head(out)
    contrastive_loss, acc = contrastive_loss_func(out, criterion, batch_size, n_views)

    return contrastive_loss , acc



