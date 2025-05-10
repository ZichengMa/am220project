import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

def get_cifar10(batch_size=128, val_split=0.1):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
  ])

  train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
  test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

  # Split train dataset into training and validation sets
  val_size = int(len(train_dataset) * val_split)
  train_size = len(train_dataset) - val_size
  train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

  return train_loader, val_loader, test_loader

from torchvision.datasets import MNIST

def get_mnist(batch_size=128, val_split=0.1):
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
  ])

  train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
  test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

  # Split train dataset into training and validation sets
  val_size = int(len(train_dataset) * val_split)
  train_size = len(train_dataset) - val_size
  train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

  return train_loader, val_loader, test_loader
