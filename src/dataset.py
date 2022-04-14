



def corrupt_dataset(clean_dataset, mode, dst_path):
    '''
    clean_dataset: pytorch object
    mode:     string, used to specify "l2, l1, ood, or l_infinity"
    dst_path: string, the destination folder, the structure should be the same as 
    # perform adversarial attack on the "test" dataset
    folder structure:
        dst_path
        |--L2 - modified images
        |--L1 - modified images
        |--Linfty - modified images
    '''
    

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
