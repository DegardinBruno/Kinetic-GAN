import numpy as np, os, re, sys, pickle
from feeder.cgan_feeder import Feeder
from scipy.ndimage import gaussian_filter1d



root_syn_data   = '/home/degar/DATASETS/st-gcn/NTU/xview-syn/filtered/train_syn_data.npy'
root_syn_label  = '/home/degar/DATASETS/st-gcn/NTU/xview-syn/filtered/train_syn_label.pkl'
qtd = 100

dataset_syn  = Feeder(root_syn_data, root_syn_label, norm=False, mmap=False)

for i in range(dataset_syn.N):
    for v in range(dataset_syn.V):
            for c in range(dataset_syn.C):
                dataset_syn.data[i, c, :, v, 0] = gaussian_filter1d(dataset_syn.data[i, c, :, v, 0], 2.0)

    if i % 100 == 0:
        print(i, dataset_syn.N)



with open('/home/degar/DATASETS/st-gcn/NTU/xview-syn/filtered/train_syn_gauss_data.npy', 'wb') as npf:
    np.save(npf, dataset_syn.data)

'''with open(root_fuse_label, 'wb') as f:
    pickle.dump(fuse_labels, f)'''

