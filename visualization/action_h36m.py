import sys, argparse
import numpy as np, os
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

sys.path.append(".")

from utils.general import check_runs

out = check_runs('synthetic')
if not os.path.exists(out): os.makedirs(out)

def gaussian_filter(data):
    T, V, C = data.shape

    for v in range(V):
        for c in range(C):
            data[:, v, c] = gaussian_filter1d(data[:, v, c],opt.sigma)

    return data


lcolor = "#A7ABB0"
rcolor = "#2E477D"

I  = np.array([0,1,2,0,4,5,0, 7,8,8,10,11,8,13,14]) # start points
J  = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) # end points
LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to generated samples")
parser.add_argument("--indexes", nargs='+', type=int, default=-1, help="FIVE sample's index")
parser.add_argument("--time", type=int, default=64, help="Temporal size") 
parser.add_argument("--joints", type=int, default=16, help="Number of joints")
parser.add_argument("--sigma", type=float, default=0, help="Gaussian filter's sigma")
parser.add_argument("--norm", action='store_true', help="Normalize values")
opt = parser.parse_args()
print(opt)

config_file = open(os.path.join(out,"config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
config_file.close()

data = np.load(opt.path, mmap_mode='r')
if opt.norm:
    data = (2 * ((data-data.min())/(data.max() - data.min())) - 1)

print('Data shape', data.shape)

data_numpy = np.array([np.transpose(data[index,:,:opt.time,:opt.joints], (1, 2, 0)) for index in opt.indexes])

if opt.sigma != 0:
    data_numpy = np.array([gaussian_filter(d) for d in data_numpy])

print(data_numpy.shape)
# Hip to 0 like other methods
tmp = []
for d in data_numpy:
    tmp = d[:,0,:]
    z = np.zeros((tmp.shape[0], tmp.shape[1] ))
    d[:,0,:] = z

print(data_numpy.max())
print(data_numpy.min())

fig, ax = plt.subplots()


data_numpy[1,:,:,0] = data_numpy[1,:,:,0]+1
data_numpy[2,:,:,0] = data_numpy[2,:,:,0]-1

data_numpy[3,:,:,0] = data_numpy[3,:,:,0]+2
data_numpy[4,:,:,0] = data_numpy[4,:,:,0]-2

print(data_numpy.shape)

for frame_idx in range(data_numpy.shape[1]):
    plt.cla()
    ax.set_title("Frame: {}".format(frame_idx))

    ax.set_xlim([-2.5, 2.5])  # Length 5
    ax.set_ylim([-1, 1])

    for data in data_numpy:

        x = data[frame_idx, :, 0]
        y = data[frame_idx, :, 1]

        for i in range( len(I) ):
            x_plot = [x[I[i]], x[J[i]]]
            y_plot = [y[I[i]], y[J[i]]]
            ax.plot(x_plot, y_plot, lw=2, c=lcolor if LR[i] else rcolor)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()

        
    plt.savefig(os.path.join(out,"frame_"+str(frame_idx)+".png"))
    print("The {} frame 2d skeleton......".format(frame_idx))
