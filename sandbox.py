import numpy as np, os, re, sys
from feeder.cgan_feeder import Feeder
from collections import Counter

root_data  = "/home/degar/DATASETS/st-gcn/NTU/xview/train_data.npy"
root_label = "/home/degar/DATASETS/st-gcn/NTU/xview/train_label.pkl"
           
classes = [0,1,2,3,4,5,]
new_labels = [0,0,0,0,0,1,1,1,1,1,]

tmp     = Counter(new_labels)
classes = [i for i in classes if tmp[i]<=4]

print(classes)