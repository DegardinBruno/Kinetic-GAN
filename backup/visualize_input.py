import numpy as np
import cv2
from PIL import ImageColor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    center_joint = data[0, :, 0, :]

    center_jointx = np.mean(center_joint[:, 0])
    center_jointy = np.mean(center_joint[:, 1])
    center_jointz = np.mean(center_joint[:, 2])

    center = np.array([center_jointx, center_jointy, center_jointz])
    data = data - center

    return data


'''                #   belly   chest    neck     head    lshoulder
neighbor_edge =   [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                # lupperarm lforearm lwrist rshoulder rupperarm
                    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                #   rforearm  rwrist    lhip     lfemur    ltibia 
                    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                #   lfoot     rhip     rfemur    rtibia    rfoot
                    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                #   lfinger   lhand    rfinger   rhand 
                    (22, 23), (23, 8), (24, 25), (25, 12)]

darkred1, darkred2, darkred3 = '#7e0008', '#580005', '#320003'
green, darkgreen, darkgreen1, darkgreen2 = '#17695b', '#125a58', '#103f49', '#183449'
lightblue1, lightblue2, lightblue3, blue = '#b6d5eb', '#7fbadc', '#54a5d5', '#3892c6'
orange, darkorange = '#f79a39', '#e64242'

#honey, darkgreen = '#FCBF5D', '#007e3d', 
#red, blue, orange, green, darkblue = '#FD0010', '#00B2EE', '#FF631C', '#00FD7B', '#0C68F5'


color_edge =   [darkred1, darkred1, darkred2, darkred3, green, 
                darkgreen, darkgreen1, darkgreen2, lightblue1, lightblue2, 
                lightblue3, blue, green, darkgreen, darkgreen1, 
                orange, lightblue1, lightblue2, lightblue3, darkorange, 
                orange, orange, darkorange, darkorange]
'''



trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]


root_data = '/home/degardin/DATASET/REGINA/SSMs/NTU/xview/train_NTU_xview_skeleton_joint.npy'
data = np.load(root_data, mmap_mode='r')


data_numpy = np.transpose(data[10], (3, 1, 2, 0))
data_numpy = rotation(data_numpy, 0,50)
data_numpy = normal_skeleton(data_numpy)


print(data_numpy.shape)
M, T, V, _ = data_numpy.shape
init_horizon=-45
init_vertical=20


fig = plt.figure()
ax = Axes3D(fig)

ax.view_init(init_vertical, init_horizon)

for frame_idx in range(data_numpy.shape[1]):

    plt.cla()
    plt.title("Frame: {}".format(frame_idx))

    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([0, 1.8])

    x = data_numpy[0, frame_idx, :, 0]
    y = data_numpy[0, frame_idx, :, 1]
    z = data_numpy[0, frame_idx, :, 2]

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

    
    plt.savefig("visualization/zau_"+str(frame_idx)+".png")
    print("The {} frame 3d skeleton......".format(frame_idx))

    ax.set_facecolor('none')