import numpy as np
import os, re, random, sys
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import matplotlib.pyplot as plt

sys.path.append(".")

from utils import general
from feeder.cgan_feeder import Feeder


test = general.load('cgan-graph', 'plot_loss', run_id=-1)


d_loss = np.concatenate(test['d_loss'])
g_loss = np.concatenate(test['g_loss'])


d_loss = np.array(np.split(d_loss, int(len(d_loss)/2352)))
g_loss = np.array(np.split(g_loss, int(len(g_loss)/2352)))


d_loss = [np.mean(loss) for loss in d_loss]
g_loss = [np.mean(loss) for loss in g_loss]


x_iter = np.arange(0,len(d_loss),1)


plt.clf()
plt.plot(x_iter, d_loss, color='blue', linewidth=1, label='D loss', alpha=0.6)
plt.plot(x_iter, g_loss, color='red', linewidth=1, label='G loss', alpha=0.6)


plt.title('G and D optimized at every batch')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()