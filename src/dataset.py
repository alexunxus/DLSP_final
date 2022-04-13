


class CleanDataset:

    def __init__(self):
        pass

    def __getitem__(self, idx):
        '''get clean training/validation/test image from cifar10 dataset'''
        pass

    def __len__(self):
        pass


class AdversarialDataset(CleanDataset):

    def __init__(self, mode = 'l2'):
        super().__init__()

        if mode not in ['l1', 'l2', 'linf', 'ood']:
            raise ValueError(f"Unkonwn mode : {mode}")
        self.mode = mode

    def __getitem__(self, idx):
        '''return an adversarially attacked image'''
        pass
