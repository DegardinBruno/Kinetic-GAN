import numpy as np
import os, re, random
import pickle
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import matplotlib.pyplot as plt

from utils import general
from feeder.cgan_feeder import Feeder



bn = nn.BatchNorm1d(64, 0.8)
print(bn.eps)
print(bn.momentum)