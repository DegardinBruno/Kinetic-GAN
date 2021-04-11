import numpy as np, os, re, sys
from feeder.cgan_feeder import Feeder


root_data = "/home/degar/DATASETS/st-gcn/NTU/xview/train_data.npy"
root_label = "/home/degar/DATASETS/st-gcn/NTU/xview/train_label.pkl"

dataset = Feeder(root_data, root_label)

print(dataset.max, dataset.min)

test = np.concatenate((np.random.uniform(-1,1,99), [1,-1]))

test = (test - test.min()) / (test.max() - test.min())


test = test*(dataset.max - dataset.min) + dataset.min
print(test)