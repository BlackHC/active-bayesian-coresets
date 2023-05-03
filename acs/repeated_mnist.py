from copy import deepcopy
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# Cell


def create_repeated_MNIST_dataset(*, num_repetitions: int = 3, seed: int = 222, add_noise: bool = True):
    # num_classes = 10, input_size = 28

    train_dataset = datasets.MNIST("data", train=True, download=True)
    X_train, Y_train = train_dataset.data, train_dataset.targets

    if num_repetitions > 1:
        X_train = torch.concat([X_train] * num_repetitions)
        Y_train = torch.concat([Y_train] * num_repetitions)

    if add_noise:
        dataset_noise = torch.empty((len(X_train), 28, 28), dtype=torch.float32).normal_(0.0, 0.1, generator=torch.Generator().manual_seed(seed))

        X_train = X_train.float()
        for idx in range(len(X_train)):
            X_train[idx] += dataset_noise[idx]


    test_dataset = datasets.MNIST("data", train=False)

    return X_train, Y_train, test_dataset


def create_MNIST_dataset():
    return create_repeated_MNIST_dataset(num_repetitions=1, seed = 222, add_noise=False)


def get_targets(dataset):
    """Get the targets of a dataset without any target transforms.
    This supports subsets and other derivative datasets."""
    if isinstance(dataset, data.Subset):
        targets = get_targets(dataset.dataset)
        return torch.as_tensor(targets)[dataset.indices]
    if isinstance(dataset, data.ConcatDataset):
        return torch.cat([get_targets(sub_dataset) for sub_dataset in dataset.datasets])

    return torch.as_tensor(dataset.targets)


class LeNetv2(nn.Module):
    
    def __init__(self):
        super(LeNetv2, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),

        )
        self.pretrained = False
        
    def forward(self,x): 
        x=self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return x