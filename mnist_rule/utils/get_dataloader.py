from torch.utils.data import DataLoader
from dataset.singlebitDataset import MnistSinglebit
from dataset.multiplebitDataset import MnistMultiplebit
from dataset.comnistDataset import CoMNIST
from torchvision import transforms, datasets


def get_tarin_loader(dataset, path, batch_size, device):
    if dataset == 'singlebit':
        dataset = MnistSinglebit(path, device=device)
    elif dataset == 'multiplebit':
        dataset = MnistMultiplebit(path, device=device)
    elif dataset == 'comnist':
        dataset = CoMNIST(path, device=device)
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
    else:
        raise 'Unknwon dataset'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def get_test_loader(dataset, path, batch_size, device):
    if dataset == 'singlebit':
        dataset = MnistSinglebit(path, device=device, training=False)
    elif dataset == 'multiplebit':
        dataset = MnistMultiplebit(path, device=device, training=False)
    elif dataset == 'comnist':
        dataset = CoMNIST(path, device=device, training=False)
    elif dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = datasets.MNIST(path, train=False, download=True, transform=transform)
    else:
        raise 'Unknwon dataset'
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
