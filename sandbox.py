import numpy as np
import os, re, random
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import matplotlib.pyplot as plt

from utils.general import *
from feeder.cgan_feeder import Feeder

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

data_path = '/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_data.npy'
label_path = '/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_label.pkl'

dataset = Feeder(data_path, label_path)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4
)

# get some random training images
dataiter = iter(dataloader)
images, labels = dataiter.next()

print(torch.max(images), torch.min(images))