import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


trunk_joints = [0, 1, 20, 2, 3]
arm_joints = [23, 24, 11, 10, 9, 8, 20, 4, 5, 6, 7, 22, 21]
leg_joints = [19, 18, 17, 16, 0, 12, 13, 14, 15]
body = [trunk_joints, arm_joints, leg_joints]


# Show 3D Skeleton with Axes3D for NTU RGB+D
class Draw3DSkeleton(object):

    def __init__(self, file, save_path=None, init_horizon=-45,
                 init_vertical=20, x_rotation=None,
                 y_rotation=None, pause_step=0.2):

        self.file = file
        self.save_path = save_path

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.xyz = self.read_xyz(self.file)

        self.init_horizon = init_horizon
        self.init_vertical = init_vertical

        self.x_rotation = x_rotation
        self.y_rotation = y_rotation

        self._pause_step = pause_step

    def _read_skeleton(self, file):
        with open(file, 'r') as f:
            skeleton_sequence = {}
            skeleton_sequence['numFrame'] = int(f.readline())
            skeleton_sequence['frameInfo'] = []
            for t in range(skeleton_sequence['numFrame']):
                frame_info = {}
                frame_info['numBody'] = int(f.readline())
                frame_info['bodyInfo'] = []
                for m in range(frame_info['numBody']):
                    body_info = {}
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key, f.readline().split())
                    }
                    body_info['numJoint'] = int(f.readline())
                    body_info['jointInfo'] = []
                    for v in range(body_info['numJoint']):
                        joint_info_key = [
                            'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                            'orientationW', 'orientationX', 'orientationY',
                            'orientationZ', 'trackingState'
                        ]
                        joint_info = {
                            k: float(v)
                            for k, v in zip(joint_info_key, f.readline().split())
                        }
                        body_info['jointInfo'].append(joint_info)
                    frame_info['bodyInfo'].append(body_info)
                skeleton_sequence['frameInfo'].append(frame_info)
        return skeleton_sequence

    def read_xyz(self, file, max_body=2, num_joint=25):
        seq_info = self._read_skeleton(file)
        data = np.zeros((3, seq_info['numFrame'], num_joint, max_body))  # (3,frame_nums,25 2)
        for n, f in enumerate(seq_info['frameInfo']):
            for m, b in enumerate(f['bodyInfo']):
                for j, v in enumerate(b['jointInfo']):
                    if m < max_body and j < num_joint:
                        data[:, n, j, m] = [v['x'], v['y'], v['z']]
                    else:
                        pass
        return data

    def _normal_skeleton(self, data):
        #  use as center joint
        center_joint = data[0, :, 0, :]

        center_jointx = np.mean(center_joint[:, 0])
        center_jointy = np.mean(center_joint[:, 1])
        center_jointz = np.mean(center_joint[:, 2])

        center = np.array([center_jointx, center_jointy, center_jointz])
        data = data - center

        return data

    def _rotation(self, data, alpha=0, beta=0):
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

    def visual_skeleton(self):
        fig = plt.figure()
        ax = Axes3D(fig)

        ax.view_init(self.init_vertical, self.init_horizon)
        plt.ion()

        data = np.transpose(self.xyz, (3, 1, 2, 0))

        # data rotation
        if (self.x_rotation is not None) or (self.y_rotation is not None):

            if self.x_rotation > 180 or self.y_rotation > 180:
                raise Exception("rotation angle should be less than 180.")

            else:
                data = self._rotation(data, self.x_rotation, self.y_rotation)

        # data normalization
        data = self._normal_skeleton(data)

        # show every frame 3d skeleton
        for frame_idx in range(data.shape[1]):

            plt.cla()
            plt.title("Frame: {}".format(frame_idx))

            ax.set_xlim3d([-1, 1])
            ax.set_ylim3d([-1, 1])
            ax.set_zlim3d([-0.8, 0.8])

            x = data[0, frame_idx, :, 0]
            y = data[0, frame_idx, :, 1]
            z = data[0, frame_idx, :, 2]

            for part in body:
                x_plot = x[part]
                y_plot = y[part]
                z_plot = z[part]
                ax.plot(x_plot, z_plot, y_plot, color='b', marker='o', markerfacecolor='r')

            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')

            if self.save_path is not None:
                save_pth = os.path.join(self.save_path, '{}.png'.format(frame_idx))
                plt.savefig(save_pth)
            print("The {} frame 3d skeleton......".format(frame_idx))

            ax.set_facecolor('none')
            plt.pause(self._pause_step)

        plt.ioff()
        ax.axis('off')
        plt.show()


if __name__ == '__main__':
    # test sample

    root_data = '/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xsub/val_data.npy'
    data = np.load(root_data, mmap_mode='r')
    sk = Draw3DSkeleton("./skimg/sk/S001C001P004R002A008.skeleton", './a08')
    sk.visual_skeleton()