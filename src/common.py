from collections import OrderedDict, namedtuple
import numpy as np
import torch

def trim_dict(state_dict, prefix = "module."):
    return OrderedDict({key.replace(prefix, ""): val for key, val in state_dict.items()})

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target])

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        for choices, f in zip(self.choices, self.transforms):
            args = {k: v[index] for (k, v) in choices.items()}
            data = f(data, **args)
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            options = t.options(x_shape)
            x_shape = t.output_shape(x_shape) if hasattr(t, 'output_shape') else x_shape
            self.choices.append({k: np.random.choice(v, size=N) for (k, v) in options.items()})


class Crop(namedtuple('Crop', ('h', 'w'))):
    def __call__(self, x, x0, y0):
        return x[:, y0:y0 + self.h, x0:x0 + self.w]

    def options(self, x_shape):
        C, H, W = x_shape
        return {'x0': range(W + 1 - self.w), 'y0': range(H + 1 - self.h)}

    def output_shape(self, x_shape):
        C, H, W = x_shape
        return (C, self.h, self.w)


class FlipLR(namedtuple('FlipLR', ())):
    def __call__(self, x, choice):
        return x[:, :, ::-1].copy() if choice else x

    def options(self, x_shape):
        return {'choice': [True, False]}

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(self.device).float(), 'target': y.to(self.device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)
    