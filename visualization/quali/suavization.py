import os, numpy as np
from scipy.ndimage import gaussian_filter1d

sigma=1

def gaussian_filter(data):
    C, T, V, _ = data.shape

    for v in range(V):
        for c in range(C):
            data[c, :, v, 0] = gaussian_filter1d(data[c, :, v, 0], sigma)

    return data

root = 'runs/kinetic-gan/exp2/actions/59_30_trunc0.95_stochastic_gen_data.npy'
dest = 'runs/kinetic-gan/exp2/actions/59_30_trunc0.95_stochastic_gen_data_suav.npy'

data = np.load(root)



data = np.array([gaussian_filter(d) for d in data])
print(data.shape)


with open(dest, 'wb') as npf:
    np.save(npf, data)

