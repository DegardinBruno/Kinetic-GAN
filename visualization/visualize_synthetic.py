import sys
sys.path.append(".")


import numpy as np, os
import cv2
from PIL import ImageColor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.general import check_runs


out        = check_runs('synthetic')
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


trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]


root_data = '/home/socialab/Desktop/PhD/Projects/Graph_GAN/runs/dcgan-graph/exp3/images/2000.npy'
data = np.load(root_data, mmap_mode='r')

print(data.shape)


data_numpy = np.transpose(data[1,:,:,:25], (1, 2, 0))
data_numpy = rotation(data_numpy, 0,50)
data_numpy = normal_skeleton(data_numpy)


print(data_numpy.shape)
T, V, _ = data_numpy.shape
init_horizon=-45
init_vertical=20


fig = plt.figure()
ax = Axes3D(fig)

ax.view_init(init_vertical, init_horizon)

for frame_idx in range(data_numpy.shape[0]):

    plt.cla()
    plt.title("Frame: {}".format(frame_idx))

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([0, 1.8])

    x = data_numpy[frame_idx, :, 0]
    y = data_numpy[frame_idx, :, 1]
    z = data_numpy[frame_idx, :, 2]

    if x[0] == x[1] == x[2]:
        break

    for part in body:
        x_plot = x[part]
        y_plot = y[part]
        z_plot = z[part]
        ax.plot(x_plot, z_plot, y_plot, color='b', marker='o', markerfacecolor='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    
    plt.savefig(os.path.join(out,"zau_"+str(frame_idx)+".png"))
    print("The {} frame 3d skeleton......".format(frame_idx))

    ax.set_facecolor('none')