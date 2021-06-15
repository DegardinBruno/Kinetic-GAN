import numpy as np
import torch
import argparse
import os,sys

sys.path.append(".")

from feeder.feeder import Feeder
from utils import general


out        = general.check_runs('mmd-actions')
if not os.path.exists(out): os.makedirs(out)

class MMD:  # class to compute MMD between two distributions

    def __init__(self, mode,use_torch):
        #  mode: 'avg' or 'joint', 'avg' computes the frame-average MMD, 'joint' regard a sequence as joint distribution
        self.mode = mode
        self.use_torch = use_torch

    def reset(self, new_mode):  # reset the status of MMD
        self.mode = new_mode

    def rkhs_mmd(self, samples_1, samples_2, bandwidth):  # two given sample groups, shape (N*dim)
        if self.use_torch:
            m, dim = samples_1.size()
            n, _ = samples_2.size()
        else:
            m, dim = np.shape(samples_1)
            n, _ = np.shape(samples_2)

        def rbf_kernel(z_1, z_2, bandwidth):
            if self.use_torch:
                z_1_expand = z_1.unsqueeze(1)
                dist_square = ((z_1_expand - z_2).pow(2.)).sum(-1)
                kernel_matrix = (-dist_square/bandwidth).exp()
            else:
                z_1_expand = np.expand_dims(z_1, axis=1)
                dist_square = np.sum((z_1_expand - z_2)**2, axis=-1)
                kernel_matrix = np.exp(-dist_square/bandwidth)
            return kernel_matrix
        kxx = rbf_kernel(samples_1, samples_1, bandwidth)
        kyy = rbf_kernel(samples_2, samples_2, bandwidth)
        kxy = rbf_kernel(samples_1, samples_2, bandwidth)
        hxy = kxx + kyy - 2*kxy

        if self.use_torch:
            mmd_ = ((hxy - hxy.diag().diag()).sum()/(m*(m-1))).pow(0.5)
            del kxx, kyy, kxy, hxy
            torch.cuda.empty_cache()
            return mmd_.item()
        else:
            mmd_ = np.sqrt(np.sum(hxy - np.diag(np.diag(hxy)))/(m*(m-1)))
            return mmd_

    def compute_sequence_mmd(self, sequence_1, sequence_2, bandwidth):  # compute the mmd between sequences, shape (N*len*dim)
        if self.use_torch:
            _, seq_len, dim = sequence_1.size()
        else:
            _, seq_len, dim = np.shape(sequence_1)
        result = 0.
        if self.mode == 'avg':
            for frames in range(seq_len):
                result += self.rkhs_mmd(sequence_1[:, frames, :], sequence_2[:, frames, :], bandwidth)/seq_len
        elif self.mode == 'joint':
            if self.use_torch:
                flat_seq_1 = sequence_1.view(-1, dim*seq_len)
                flat_seq_2 = sequence_2.view(-1, dim*seq_len)
            else:
                flat_seq_1 = np.reshape(sequence_1, (-1, dim*seq_len))
                flat_seq_2 = np.reshape(sequence_2, (-1, dim*seq_len))
            result = self.rkhs_mmd(flat_seq_1, flat_seq_2, bandwidth)
        else:
            raise Exception('undefined mode')
        return result


def calcualte_mmd(gen,real,label):
    use_torch = 1
    class_num = np.shape(label)[-1]
    gen_data_list = [[] for i in range(class_num)]
    real_data_list = [[] for j in range(class_num)]
    

    mode = opt.mmd_mode
    compute_mmd = MMD(mode,use_torch)
    # for i in tqdm(range(len(gen)), total=len(gen)):
    for i in range(len(gen)):
        cl = np.argmax(label[i])
        if len(gen_data_list[cl]) < 2000:  # NOTE: joint mode can not afford to large matrix
            gen_data_list[cl].append(gen[i])
            real_data_list[cl].append(real[i])

    result_list = []
    for i in range(class_num):
        new_r = 0
        new_gen = np.asarray(gen_data_list[i])
        new_real = np.asarray(real_data_list[i])
        if use_torch:
            new_gen = torch.tensor(new_gen).cuda()
            new_real = torch.tensor(new_real).cuda()
        # for j in tqdm(range(-4, 10), total=14):

        for j in range(-4,10):
            new_new_r = compute_mmd.compute_sequence_mmd(new_gen[0], new_real[0], 10 ** j)
            if new_new_r > new_r:
                new_r = new_new_r
        # print(new_r)
        result_list.append(new_r)
        del new_real, new_gen
        torch.cuda.empty_cache()
    result = np.mean(result_list)

    return result



parser = argparse.ArgumentParser()
parser.add_argument("--data_real", type=str, default="/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/cGC-GAN/datasets/h36m/train_data.npy", help="path to real data")
parser.add_argument("--labels_real", type=str, default="/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/cGC-GAN/datasets/h36m/train_label.pkl", help="path to real labels")
parser.add_argument("--data_fake", type=str, default="/home/socialab/Desktop/PhD/Projects/Graph_GAN/runs/cgc-gan/exp6/images/mmd-seq-0.08/10_1000_gen_data.npy", help="path to fake data")
parser.add_argument("--labels_fake", type=str, default="/home/socialab/Desktop/PhD/Projects/Graph_GAN/runs/cgc-gan/exp6/images/mmd-seq-0.08/10_1000_gen_label.pkl", help="path to real labels")
parser.add_argument("--mmd_mode", type=str, default="avg", choices=['avg', 'joint'], help="avg for dynamics and joint for whole sequence")
parser.add_argument("--t_size", type=int, default=64, help="Temporal dimension")
parser.add_argument("--dataset", type=str, default="h36m", help="dataset to evaluate")
opt = parser.parse_args()
print(opt)


dataset_real = Feeder(opt.data_real, opt.labels_real, norm=True, dataset=opt.dataset)   # Normalize to [-1, 1]
dataset_fake = Feeder(opt.data_fake, opt.labels_fake, norm=False, dataset=opt.dataset)  # Already normalized

fake_actions_batch = []
real_actions_batch = []
label_batch = []
fake_label_batch = []
labels =  np.arange((10 if opt.dataset=='h36m' else 60)) # If dataset chosen is NTU, you can select the classes for fair evaluation 
i=0
c = 0
while c < len(labels):
    if dataset_real[i][1] == labels[c]:
        real_actions_batch.append(dataset_real[i][0][:,:opt.t_size,:])
        label_batch.append(dataset_real[i][1])
        if len(label_batch) == 100*c+100:
            c+=1
            i = 0
    i+=1

i = 0
c = 0
while c<len(labels):
    if dataset_fake[i][1] == labels[c]:
        fake_actions_batch.append(dataset_fake[i][0][:,:opt.t_size,:])
        fake_label_batch.append(dataset_fake[i][1])

        if len(fake_label_batch) == 100*c+100:
            c+=1
            i = 0

    i+=1

assert fake_label_batch == label_batch

label_batch = np.array(label_batch)

label_batch = np.nonzero(label_batch[:, None] == labels)[1]

print(np.array(real_actions_batch).shape,'real')
print(np.array(fake_actions_batch).shape)

b = np.zeros((label_batch.size, label_batch.max()+1))
b[np.arange(label_batch.size),label_batch] = 1
label_batch = b

print(label_batch.shape)
print(np.asarray(fake_actions_batch).max(), np.asarray(fake_actions_batch).min())
print(np.asarray(real_actions_batch).max(), np.asarray(real_actions_batch).min())

fake_actions_batch = np.asarray(fake_actions_batch).transpose(0,3,2,1)
real_actions_batch = np.asarray(real_actions_batch).transpose(0,3,2,1)


print(real_actions_batch.shape)
print(fake_actions_batch.shape)

result = calcualte_mmd(fake_actions_batch, real_actions_batch, np.asarray(label_batch))

config_file = open(os.path.join(out,"config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt) + '\n'+'MMD_'+ str(opt.mmd_mode) +': ' + str(result))
config_file.close()

print(result)