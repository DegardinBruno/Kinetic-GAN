import numpy as np
import os, re, random
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt

from utils.general import *

data = np.array([1,1,1,2,2,2,2,3,3,4,5])
label = np.array([1,2,3,4,5,6,7,8,9,10])

classes = np.array([5,3,4,2])

tmp = label[np.where(np.isin(label, classes))]
data  = data[np.where(np.isin(label, classes))]
label = np.nonzero(tmp[:, None] == classes)[1]

print(data)
print(label)
print(len(label))