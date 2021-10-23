import bpy
import numpy as np
import mathutils
from mathutils import Quaternion, Matrix


def rotation(data, alpha=0, beta=0, charlie=0):
        # rotate the skeleton around x-y axis
        r_alpha = alpha * np.pi / 180
        r_beta = beta * np.pi / 180
        r_charlie = charlie * np.pi / 180

        rx = np.array([[1, 0, 0],
                       [0, np.cos(r_alpha), -1 * np.sin(r_alpha)],
                       [0, np.sin(r_alpha), np.cos(r_alpha)]]
                      )

        ry = np.array([
            [np.cos(r_beta), 0, np.sin(r_beta)],
            [0, 1, 0],
            [-1 * np.sin(r_beta), 0, np.cos(r_beta)],
        ])


        rz = np.array([
            [np.cos(r_charlie), -1 * np.sin(r_charlie), 0],
            [np.sin(r_charlie), np.cos(r_charlie), 0],
            [0, 0, 1]
        ])

        r = ry.dot(rx).dot(rz)
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


data_o = np.load('/home/degar/PhD/Projects/Kinetic-GAN/runs/kinetic-gan/exp1/actions/jump/26_50_trunc0.95_gen_data.npy', mmap_mode='r')
skl = 25  # 5


data = data_o[skl,:,:,:,0]*16
data = np.transpose(data, (1,2,0))
data = np.array(rotation(data, -110, 0,0))  # - backwards, + forward; Rotate on x-axis and y-axis to align visualization
data = np.array(normal_skeleton(data))  # Align to zero, comment if no need



data[:,16,0] = data[:,16,0] +1
data[:,12,0] = data[:,12,0] -1
data[:,8,0] = data[:,8,0] + 1
data[:,4,0] = data[:,4,0] - 1


print(data.shape)

bone_order = {'Hips':1, 
              'Thorax':20,
              'Neck':2, 'Head':3,
              'Hips.R':16, 'Leg.R':17, 'Lowerleg.R':18, 'Foot.R':19,
              'Hips.L':12, 'Leg.L':13, 'Lowerleg.L':14, 'Foot.L':15,
              'Shoulder.R':8, 'Arm.R':9, 'Forearm.R':10, 'Hand.R':11, 'Thumb.R':24, 'Finger.R':23,
              'Shoulder.L':4, 'Arm.L':5, 'Forearm.L':6, 'Hand.L':7, 'Thumb.L':22, 'Finger.L':21}

arm = bpy.data.objects['Armature'] 
s=bpy.context.scene

              
print(len(bone_order))

print(bone_order.keys())
bpy.context.active_object.animation_data_clear()

for t_f in range(len(data)):

        #t_f = 47
        for i, key in enumerate(bone_order.keys()):
            print(i, key)
            bone = arm.pose.bones[key]
            
            new_loc = mathutils.Vector(data[t_f,bone_order[key],:])
            
            
            print(new_loc[0], new_loc[1], new_loc[2])
            print(bone.location[0],bone.location[2],bone.location[1])
            
            
            if bone_order[key] == 1 or bone_order[key]==3 or bone_order[key]==2 or bone_order[key]==19 or bone_order[key]==15:
                bpy.ops.object.mode_set(mode='POSE')
                
                bone.location[0] = new_loc[0]
                bone.location[1] = new_loc[2]
                bone.location[2] = -new_loc[1]
                
                
                bone.keyframe_insert(data_path='rotation_euler',frame=t_f+1)
                bone.keyframe_insert(data_path='rotation_quaternion',frame=t_f+1)
                bone.keyframe_insert(data_path='scale',frame=t_f+1)
                bone.keyframe_insert(data_path='location',frame=t_f+1)
            
            else:
                
                bpy.ops.object.mode_set(mode='POSE')
                

                v = new_loc - bone.head
                bv = bone.tail - bone.head

                rd = bv.rotation_difference(v)

                M = (
                    Matrix.Translation(bone.head) @
                    rd.to_matrix().to_4x4() @
                    Matrix.Translation(-bone.head)
                    )
                bone.matrix = M @ bone.matrix

                
                bone.keyframe_insert(data_path='rotation_euler',frame=t_f+1)
                bone.keyframe_insert(data_path='rotation_quaternion',frame=t_f+1)
                bone.keyframe_insert(data_path='scale',frame=t_f+1)
                bone.keyframe_insert(data_path='location',frame=t_f+1)


        bpy.context.scene.frame_set(t_f+1)
        s.render.filepath = "/home/degar/Desktop/Docs/Kinetic-GAN/videos/jump_3/"+str(t_f)+"_"+str(skl)+".png"
        bpy.ops.render.render( #{'dict': "override"},
                                  #'INVOKE_DEFAULT',  
                                  False,            # undo support
                                  animation=False, 
                                  write_still=True
                                 )
          
    