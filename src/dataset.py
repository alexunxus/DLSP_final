from torchvision.datasets import CIFAR10
from torchvision import transforms

from sklearn.model_selection import train_test_split

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
    
# preprocess data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]),
}

random_state = 100

def split_dataset():
    data = CIFAR10("./data/cifar10/", train=True  ,download=True, transform=data_transforms['train'])
    x_train, y_train, x_valid, y_valid = train_test_split(data, test_size=0.2, random_state=random_state)
    return x_train, y_train, x_valid, y_valid

class CleanDataset:

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        '''get clean training/validation/test image from cifar10 dataset'''
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.X.shape[0]


class AdversarialDataset(CleanDataset):

    def __init__(self, mode = 'l2'):
        super().__init__()

        if mode not in ['l1', 'l2', 'linf', 'ood']:
            raise ValueError(f"Unkonwn mode : {mode}")
        self.mode = mode

    def __getitem__(self, idx):
        '''return an adversarially attacked image'''
        pass


if __name__ == '__main__':
    # corrupt dataset here
    pass