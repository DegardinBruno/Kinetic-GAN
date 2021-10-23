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
    """ Feeder for skeleton-based action synthesis
    Arguments:
        data_path: the path to '.npy' data, the shape of real data should be (N, C, T, V, M) for NTU and (N,C,T,V) for h36m
        label_path: the path to label
    """

    def __init__(self,
                 data_path,
                 label_path,
                 classes=None,
                 norm=True,
                 dataset='ntu',
                 mmap=True):
        self.data_path  = data_path
        self.label_path = label_path
        self.classes    = classes  # If we want to use only specified classes (NOT USED IN KINETIC-GAN PAPER)
        self.norm       = norm
        self.dataset    = dataset
        self.load_data(mmap)


    def load_data(self, mmap):
        # data: N C T V M

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)
        self.label = np.array(self.label, dtype=int)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.max, self.min = self.data.max(), self.data.min()

        if self.classes is not None:
            tmp = self.label[np.where(np.isin(self.label, self.classes))]
            self.data  = self.data[np.where(np.isin(self.label, self.classes))]
            self.label = np.nonzero(tmp[:, None] == self.classes)[1]

        if self.dataset=='ntu':
            self.N, self.C, self.T, self.V, self.M = self.data.shape
        else:
            self.N, self.C, self.T, self.V = self.data.shape

        print(self.data.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index,:,:,:,0]) if self.dataset=='ntu' else np.array(self.data[index])
        data_numpy = 2 * ((data_numpy-self.min)/(self.max - self.min)) - 1 if self.norm else data_numpy
        label = self.label[index]
        
        return data_numpy, label

    