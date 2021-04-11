import numpy as np, os, re, sys, pickle
from feeder.cgan_feeder import Feeder
from collections import Counter



root_real_data  = "/home/degar/DATASETS/st-gcn/NTU/xview/train_data.npy"
root_real_label = "/home/degar/DATASETS/st-gcn/NTU/xview/train_label.pkl"
root_syn_data   = '/home/degar/DATASETS/st-gcn/NTU/xview-syn/filtered/train_syn_gauss_data.npy'
root_syn_label  = '/home/degar/DATASETS/st-gcn/NTU/xview-syn/filtered/train_syn_gauss_label.pkl'
root_fuse_data  = '/home/degar/DATASETS/st-gcn/NTU/xview-syn/filtered/train_fuse_gauss_data.npy'
root_fuse_label  = '/home/degar/DATASETS/st-gcn/NTU/xview-syn/filtered/train_fuse_gauss_label.pkl'
qtd = 100

dataset_real = Feeder(root_real_data, root_real_label, norm=False)
dataset_syn  = Feeder(root_syn_data, root_syn_label, norm=False)

print(dataset_real.data.shape)
print(dataset_syn.data.shape)

labels = Counter(dataset_syn.label)
labels = np.array([l for l in labels])

new_syn_data  = []
new_syn_label = []
new_syn_name  = []

for l in labels:
    l_i = dataset_syn.label[np.where(dataset_syn.label == l)[0]]
    print(l, len(l_i))

    for i in l_i[:qtd]:
        new_syn_data  = np.concatenate((new_syn_data, np.expand_dims(dataset_syn.data[i], 0)), axis=0) if len(new_syn_data) else np.expand_dims(dataset_syn.data[i], 0)
        new_syn_label = np.concatenate((new_syn_label, np.expand_dims(dataset_syn.label[i], 0)), axis=0) if len(new_syn_label) else np.expand_dims(dataset_syn.label[i], 0)
        new_syn_name  = np.concatenate((new_syn_name, np.expand_dims(dataset_syn.sample_name[i], 0)), axis=0) if len(new_syn_name) else np.expand_dims(dataset_syn.sample_name[i], 0)


print(new_syn_data.shape)
print(new_syn_label.shape)
print(new_syn_name.shape)

fuse_data        = np.concatenate((dataset_real.data, new_syn_data), axis=0)
fuse_label       = np.concatenate((dataset_real.label, new_syn_label), axis=0)
fuse_sample_name = np.concatenate((dataset_real.sample_name, new_syn_name), axis=0)

fuse_labels = np.concatenate((np.expand_dims(fuse_sample_name, 0), np.expand_dims(fuse_label, 0)), axis=0)

print(fuse_data.shape)
print(fuse_labels.shape)

with open(root_fuse_data, 'wb') as npf:
    np.save(npf, fuse_data)

with open(root_fuse_label, 'wb') as f:
    pickle.dump(fuse_labels, f)

