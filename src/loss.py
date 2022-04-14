import torch
import torch.functional as F


def constrastive_loss_func(contrastive_head, criterion, batchsize, n_views, temperature=0.2):
    """ 
        Loss function for contrastive SSL learning
        contrastive_head: (B, feature_size) Tensor
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