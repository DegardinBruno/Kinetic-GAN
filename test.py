import numpy as np


root_data = '/home/socialab/Desktop/PhD/Projects/Graph-GAN/images/3200.npy'
data = np.load(root_data, mmap_mode='r')

print(data[0].shape)