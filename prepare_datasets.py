from torchvision import datasets, transforms
import torch
import os


def get_datasets(dataset_name, data_path):
    """ returns train and test datasets """
    round_dict = {'fmnist': 50, 
                  'cifar10': 100,
                  'cifar100': 100,
                  'tiny': 50
                  }
    train_dataset, test_dataset = None, None

    if dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
        num_target = 10
        
    elif dataset_name == 'svhn':
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        transform = transforms.Compose([
                    transforms.Resize((32, 32)),  # Resize to 32x32 (SVHN is already RGB and 32x32)
                    transforms.ToTensor(),  # Convert to Tensor
                    normalize,
                ])
        train_dataset = datasets.SVHN(data_path, split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(data_path, split='test', download=True, transform=transform)

        train_dataset.targets = torch.LongTensor(train_dataset.labels)
        num_target = 10

    elif dataset_name == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                                             std=[0.2675, 0.2565, 0.2761])
        transform = transforms.Compose([
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])
        valid_transform = transforms.Compose([transforms.ToTensor(),
                                              normalize])
        train_dataset = datasets.CIFAR100(data_path,
                                          train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(data_path,
                                         train=False, download=True, transform=valid_transform)
        train_dataset.targets, test_dataset.targets = torch.LongTensor(train_dataset.targets), torch.LongTensor(
            test_dataset.targets)
        num_target = 100
        
    elif dataset_name == 'fmnist':
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        transform = transforms.Compose([
                    # transforms.Resize((32, 32)),
                    transforms.ToTensor(),  
                    normalize,
                ])
        train_dataset = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

        train_dataset.targets = torch.LongTensor(train_dataset.targets)
        num_target = 10

    elif dataset_name == "tiny":
        _data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }
        _data_path = f"{data_path}/tiny-imagenet-200"
        train_dataset = datasets.ImageFolder(os.path.join(_data_path, 'train'),
                                             _data_transforms['train'])
        test_dataset = datasets.ImageFolder(os.path.join(_data_path, 'test'),
                                            _data_transforms['test'])
        train_dataset.targets = torch.tensor(train_dataset.targets)
        test_dataset.targets = torch.tensor(test_dataset.targets)
        num_target = 200
        
    return train_dataset, test_dataset, num_target, round_dict[dataset_name]