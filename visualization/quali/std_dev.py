from PIL import Image
import os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams.update({'font.size': 17})

def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)

bone_order = {'Hips':0,'Spine':1, 'Thorax':20,'Neck':2, 'Head':3,
            'Shoulder.L':8, 'Elbow.L':9, 'Wrist.L':10, 'Hand.L':11, 'Thumb.L':24, 'Finger.L':23,
            'Shoulder.R':4, 'Elbow.R':5, 'Wrist.R':6,'Hand.R':7,'Thumb.R':22, 'Finger.R':21,
            'Hips.L': 16, 'Knee.L': 17, 'Ankle.L': 18, 'Foot.L': 19,
            'Hips.R':12, 'Knee.R':13, 'Ankle.R':14, 'Foot.R':15,}


root = 'runs/kinetic-gan/exp2/actions/59_70_trunc0.95_stochastic_gen_data.npy'
dest = 'runs/kinetic-gan/exp2/actions/59_30_trunc0.95_stochastic_gen_data_suav.npy'

data = np.load(root)
data = data[:,:,51,:,0]

print(data.shape)

y = []
for bone in bone_order:
    print(data[:,:,bone_order[bone]].std())
    y.append(data[:,:,bone_order[bone]].std())


y = np.array(y)

#y = (y-y.min())/(y.max()-y.min())


fig, ax = plt.subplots(figsize=(10,4.5))

for i, bone in enumerate(bone_order):

    ax.bar(bone,y[i], color='#9E0B0F' if '.R' in bone else '#426CA6' if '.L' in bone else '#4D4D4D')
plt.setp( ax.xaxis.get_majorticklabels(), rotation=45, ha="right" )

dx = 5/72.; dy = 0/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)

for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)

for i, (bone, tick) in enumerate(zip(bone_order, ax.get_xticklabels())):
    if '.R' in bone:
        tick.set_color("#9E0B0F")
    
    if '.L' in bone:
        tick.set_color("#274063")


plt.ylabel('Standard Deviation')
c_patch = mpatches.Patch(color='#4D4D4D', label='Center')
l_patch = mpatches.Patch(color='#426CA6', label='Left')
r_patch = mpatches.Patch(color='#9E0B0F', label='Right')
plt.legend(handles=[c_patch, l_patch, r_patch], bbox_to_anchor=[0.05, 0.54],)
plt.xlim(-0.6, 24.6)
plt.tight_layout()
plt.savefig('/home/socialab/Downloads/backup/Kinetic-GAN/blender/stochastic_hist_walk.pdf')
plt.show()



'''root = "/home/socialab/Desktop/Docs/Kinetic-GAN/videos/walk_1/std_dev/"
images = [os.path.join(root, img) for img in humanSort(os.listdir(root)) if img.endswith(".png")]


all_imgs = []
for img in images:



    # load the image
    image = Image.open(img)
    # convert image to numpy array
    data = np.asarray(image)
    
    all_imgs.append(data)
    

all_imgs = np.array(all_imgs)
print(all_imgs.shape)

std_dev = np.zeros((all_imgs.shape[1:-1]))

for h in range(all_imgs.shape[1]):
    for w in range(all_imgs.shape[2]):
        std_dev[h,w] = all_imgs[:,h,w,:].std()



im = Image.fromarray(std_dev).convert('RGB')
data = np.asarray(im)
maxi, mini = data.max(), data.min()

data = np.uint8((data - mini)*255.0 / (maxi - mini))
im = Image.fromarray(data).convert('RGB')
print(im.getextrema())

im.save("std_dev_walk_53.png")'''