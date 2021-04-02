import numpy as np
import os, re, random
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import matplotlib.pyplot as plt

from utils import general
from feeder.cgan_feeder import Feeder




dataset = Feeder('/home/degar/DATASETS/st-gcn/NTU/xview/train_data.npy', '/home/degar/DATASETS/st-gcn/NTU/xview/train_label.pkl')


print(np.arange(60))


print(dataset.N)