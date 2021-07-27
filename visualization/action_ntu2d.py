import sys, argparse
import numpy as np, os, pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter1d

sys.path.append(".")

from utils.general import check_runs


out = check_runs('synthetic')
if not os.path.exists(out): os.makedirs(out)

def rotation(data, alpha=0, beta=0):
        # rotate the skeleton around x-y axis
        r_alpha = alpha * np.pi / 180
        r_beta = beta * np.pi / 180

        rx = np.array([[1, 0, 0],
                       [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
                       [0, np.sin(r_alpha), np.cos(r_alpha)]]
                      )

        ry = np.array([
            [np.cos(r_beta), 0, np.sin(r_beta)],
            [0, 1, 0],
            [-1 * np.sin(r_beta), 0, np.cos(r_beta)],
        ])

        r = ry.dot(rx)
        data = data.dot(r)

        return data


def normal_skeleton(data):
    #  use as center joint
    center_joint = data[:, 0, :]

    center_jointx = np.mean(center_joint[:, 0])
    center_jointy = np.mean(center_joint[:, 1])
    center_jointz = np.mean(center_joint[:, 2])

    center = np.array([center_jointx, center_jointy, center_jointz])
    data = data - center

    return data

def gaussian_filter(data):
    T, V, C = data.shape

    for v in range(V):
        for c in range(C):
            data[:, v, c] = gaussian_filter1d(data[:, v, c],opt.sigma)

    return data


trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]




parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to generated samples")
parser.add_argument("--labels", type=str, help="Path to generated samples")
parser.add_argument("--label", type=int, default=-1, help="Path to generated samples")
parser.add_argument("--indexes", nargs='+', type=int, default=-1, help="THREE sample's index")
parser.add_argument("--time", type=int, default=64, help="Temporal size") 
parser.add_argument("--joints", type=int, default=25, help="Number of joints")
parser.add_argument("--sigma", type=float, default=0, help="Gaussian filter's sigma")
parser.add_argument("--norm", action='store_true', help="Normalize values")

opt = parser.parse_args()
print(opt)

config_file = open(os.path.join(out,"config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
config_file.close()

data = np.load(opt.path, mmap_mode='r')
if len(data.shape) > 4:
    data = np.squeeze(data, axis=-1)

if opt.norm:
    data = (2 * ((data-data.min())/(data.max() - data.min())) - 1)

if opt.labels is not None:
    with open(opt.labels, 'rb') as f:
        _, labels = pickle.load(f)
    labels = np.array(labels)

    print('Labels shape', labels.shape)

if opt.label != -1:
    data = data[np.where(labels==opt.label)[0]]

print('Data shape', data.shape)

data_numpy = np.array([np.transpose(data[index,:,:opt.time,:opt.joints], (1, 2, 0)) for index in opt.indexes])
data_numpy[:,:,:,1] = -data_numpy[:,:,:,1]
#data_numpy = np.array([rotation(d, 0,50) for d in data_numpy])  # Rotate on x-axis and y-axis to align visualization
#data_numpy = np.array([normal_skeleton(d) for d in data_numpy])  # Align to zero, comment if no need
print(data_numpy.max())
print(data_numpy.min())
if opt.sigma != 0:
    data_numpy = np.array([gaussian_filter(d) for d in data_numpy])
print(data_numpy.max())
print(data_numpy.min())

print(data_numpy.shape)


I, T, V, _ = data_numpy.shape
init_horizon=-45
init_vertical=20


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


        for part in body:
            x_plot = x[part]
            y_plot = y[part]
            ax.plot(x_plot, y_plot, color='#2E477D', marker='o', markerfacecolor='#A7ABB0')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        
    plt.savefig(os.path.join(out,"frame_"+str(frame_idx)+".png"))
    #plt.show()
    print("The {} frame 3d skeleton......".format(frame_idx))

    ax.set_facecolor('none')