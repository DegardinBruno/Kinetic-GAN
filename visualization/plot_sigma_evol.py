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



fid_u = [296.56, 292.85, 288.85, 284.97, 281.81, 279.48, 277.87, 276.87, 276.41 ,276.36, 276.51, 276.68,276.91, 277.14, 277.32 ]
fid_f = [266.49, 263.21, 259.62, 256.13, 253.10, 250.59, 248.66, 247.26, 246.32, 245.71, 245.28, 244.96,244.81, 244.75, 244.76 ]
sigma = np.arange(1.1,2.6,0.1)

print(sigma)

plt.clf()
plt.plot(sigma, fid_u, color='red', linewidth=1, label='Unfiltered Samples', alpha=0.6)
plt.plot(sigma, fid_f, color='blue', linewidth=1, label='Filtered Samples', alpha=0.6)

plt.scatter(2.4, 244.75, marker='x', color='blue')
plt.scatter(2.0, 276.36, marker='x', color='red')


plt.title("Sigma impact over the FID score")
plt.xlabel("Sigma")
plt.ylabel("Fréchet Inception Distance")
plt.legend()
plt.grid(True)
plt.show()

