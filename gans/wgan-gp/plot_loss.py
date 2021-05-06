import numpy as np
import os, re, random, sys
import pickle
import torch
import torchvision
import torchvision.transforms as transforms
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat


test = loadmat('loss.mat')
batch_ep = 187


d_loss = np.concatenate(test['d_loss'])
g_loss = np.concatenate(test['g_loss'])


d_loss = d_loss[:int(len(d_loss)/batch_ep)*batch_ep]
g_loss = g_loss[:int(len(g_loss)/batch_ep)*batch_ep]
d_loss = np.array(np.split(d_loss, int(len(d_loss)/batch_ep)))
g_loss = np.array(np.split(g_loss, int(len(g_loss)/batch_ep)))


d_loss = [np.mean(loss) for loss in d_loss]
g_loss = [np.mean(loss) for loss in g_loss]


x_iter = np.arange(0,len(d_loss),1)


plt.clf()
plt.plot(x_iter, d_loss, color='blue', linewidth=1, label='D loss', alpha=0.6)
plt.plot(x_iter, g_loss, color='red', linewidth=1, label='G loss', alpha=0.6)


plt.title('WGAN-GP - Critic x5')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig('loss.pdf')
plt.show()