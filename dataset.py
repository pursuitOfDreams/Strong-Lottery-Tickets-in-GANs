import torchvision
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch.nn.utils.prune as prune
from torchvision import transforms

def get_default_device():
  """pick GPU if available , else GPU"""
  if torch.cuda.is_available():
    return torch.device('cuda')
  else: 
    return torch.device('cpu')

def to_device(data ,device):
  """Move tensor(s) to chosen device"""
  if isinstance(data,(list,tuple)):
    return [to_device(x,device) for x in data]
  return data.to(device, non_blocking=True)

class DeviceDataLoader():
  """Wrap a dataloader to move data to a device"""
  def __init__(self,dl,device):
    self.dl=dl
    self.device=device

  def __iter__(self):
    """Yield a batch of data after moving ot to device"""
    for b in self.dl:
      yield to_device(b,self.device)

  def __len__(self):
    """Number of Batches"""
    return len(self.dl)
  
transform = torchvision.transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,)), 
      transforms.Resize(64)
  ])
device = get_default_device()

batch_size = 64
trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
trainloader = DeviceDataLoader(trainloader, device)

testset = torchvision.datasets.MNIST(root='./data/', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
testloader = DeviceDataLoader(testloader, device)