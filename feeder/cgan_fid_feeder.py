# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from PIL import Image
from scipy.ndimage import gaussian_filter1d

# visualization
import time


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
    """

    def __init__(self,
                 data_path,
                 label_path=None,
                 classes=None,
                 norm=True,
                 sigma=1.1,
                 mmap=True):
        self.data_path  = data_path
        self.label_path = label_path
        self.classes    = classes
        self.norm       = norm
        self.sigma      = sigma
        self.load_data(mmap)


    def load_data(self, mmap):
        # data: N C V T M

        if self.label_path:
            # load label
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
            self.label = np.array(self.label)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
            #self.data = self.data[:5] if self.norm==False else self.data
        else:
            self.data = np.load(self.data_path)

        if self.classes is not None:
            tmp = self.label[np.where(np.isin(self.label, self.classes))]
            self.data  = self.data[np.where(np.isin(self.label, self.classes))]
            self.label = np.nonzero(tmp[:, None] == self.classes)[1]

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.max, self.min = self.data.max(), self.data.min()

    def __len__(self):
        return self.N

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index,:,:64,:,0])
        # data_numpy = (2 * ((data_numpy-self.min)/(self.max - self.min)) - 1) if self.norm else data_numpy
        if not self.norm and self.sigma != 0:
            for v in range(self.V):
                for c in range(self.C):
                    data_numpy[c, :, v] = gaussian_filter1d(data_numpy[c, :, v], self.sigma)

        label = self.label[index] if self.label_path else []
        

        return data_numpy, label