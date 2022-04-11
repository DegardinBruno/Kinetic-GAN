import argparse
import os
import numpy as np

from torch.autograd import Variable
import torch

from models.generator import Generator
from utils import general
from collections import Counter
import pickle


def trunc(latent, mean_size, truncation):  # Truncation trick on Z
    t = Variable(FloatTensor(np.random.normal(0, 1, (mean_size, *latent.shape[1:]))))
    m = t.mean(0, keepdim=True)

    for i,_ in enumerate(latent):
        latent[i] = m + truncation*(latent[i] - m)

    return latent


out         = general.check_runs('kinetic-gan', id=-1)
actions_out = os.path.join(out, 'actions')
if not os.path.exists(actions_out): os.makedirs(actions_out)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int,   default=10,    help="How many samples PER CLASS (each iteration of course)")
parser.add_argument("--latent_dim", type=int,   default=512,   help="dimensionality of the latent space")
parser.add_argument("--mlp_dim",    type=int,   default=4,     help="mapping network depth")
parser.add_argument("--n_classes",  type=int,   default=60,    help="number of classes for dataset")
parser.add_argument("--label",      type=int,   default=-1,    help="Sepecific label to generate, -1 for all classes")
parser.add_argument("--t_size",     type=int,   default=64,    help="size of each temporal dimension")
parser.add_argument("--v_size",     type=int,   default=25,    help="size of each spatial dimension (vertices)")
parser.add_argument("--channels",   type=int,   default=3,     help="number of channels (coordinates)")
parser.add_argument("--dataset",    type=str,   default="ntu", help="dataset")
parser.add_argument("--model",      type=str,   default="runs/kinetic-gan/exp1/models/generator_ntu_xsub_mlp4_1370000.pth", help="path to gen model")
parser.add_argument("--stochastic", action='store_true',       help="Generate/Get one sample and verify stochasticity")
parser.add_argument("--stochastic_file", type=str, default="-", help="Read one sample and verify stochasticity")
parser.add_argument("--stochastic_index", type=int, default=0, help="Sample index to get your latent point")
parser.add_argument("--gen_qtd",    type=int,   default=1000,  help="How many samples to generate per class")
parser.add_argument("--trunc",      type=float, default=0.95,   help="Truncation sigma")
parser.add_argument("--trunc_mode", type=str,   default='w',   choices=['z', 'w', '-'], help="Truncation mode (check paper for details)")
parser.add_argument("--mean_size",  type=int,   default=1000,  help="Samples to estimate mean")
opt = parser.parse_args()
print(opt)

config_file = open(os.path.join(out,"gen_config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
config_file.close()

cuda = True if torch.cuda.is_available() else False
print(cuda)

# Initialize generator 
generator     = Generator(opt.latent_dim, opt.channels, opt.n_classes, opt.t_size, mlp_dim=opt.mlp_dim, dataset=opt.dataset)

if cuda:
    generator.cuda()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Load Models
generator.load_state_dict(torch.load(opt.model), strict=False)
generator.eval()

new_imgs   = []
new_labels = []
z_s        = []

classes = np.arange(opt.n_classes) if opt.label == -1 else [opt.label]
qtd = opt.batch_size

if opt.stochastic_file!='-':
    stoch = np.load(opt.stochastic_file) 
    stoch = np.expand_dims(stoch[opt.stochastic_index], 0)
    print(stoch.shape)

if opt.stochastic:  # Generate one latent point 
    z   = Variable(FloatTensor(np.random.normal(0, 1, (1, opt.latent_dim))  if opt.stochastic_file == '-' else stoch ))
    z   = z.repeat(qtd*len(classes),1)

while(len(classes)>0):

    if not opt.stochastic: # Generate Samples if not in mode stochastic
        z         = Variable(FloatTensor(np.random.normal(0, 1, (qtd*len(classes), opt.latent_dim)))) 

    z         = trunc(z, opt.mean_size, opt.trunc) if opt.trunc_mode=='z' else z
    labels_np = np.array([num for _ in range(qtd) for num in classes])  # Generate labels
    labels    = Variable(LongTensor(labels_np))
    gen_imgs  = generator(z, labels, opt.trunc) if opt.trunc_mode == 'w' else generator(z, labels)

    new_imgs   = gen_imgs.data.cpu()  if len(new_imgs)==0 else np.concatenate((new_imgs, gen_imgs.data.cpu()), axis=0)
    new_labels = labels_np if len(new_labels)==0 else np.concatenate((new_labels, labels_np), axis=0)
    z_s        = z.cpu()  if len(z_s)==0 else np.concatenate((z_s, z.cpu()), axis=0)   
    

    tmp     = Counter(new_labels)
    classes = [i for i in classes if tmp[i]<opt.gen_qtd]

    print('---------------------------------------------------')
    print(tmp)
    print(len(new_labels), classes)


if opt.dataset == 'ntu':
    new_imgs = np.expand_dims(new_imgs, axis=-1)
    

new_labels = np.concatenate((np.expand_dims(new_labels, 0), np.expand_dims(new_labels, 0)), axis=0)  # Remove if not needed

with open(os.path.join(actions_out, str(opt.n_classes if opt.label == -1 else opt.label)+'_'+str(opt.gen_qtd)+('_trunc' + str(opt.trunc) if opt.trunc_mode!='-' else '')+('_stochastic' if opt.stochastic else '')+'_gen_data.npy'), 'wb') as npf:
    np.save(npf, new_imgs)


with open(os.path.join(actions_out, str(opt.n_classes if opt.label == -1 else opt.label)+'_'+str(opt.gen_qtd)+('_trunc' + str(opt.trunc) if opt.trunc_mode!='-' else '')+('_stochastic' if opt.stochastic else '')+'_gen_z.npy'), 'wb') as npf:
    np.save(npf, z_s)


with open(os.path.join(actions_out, str(opt.n_classes if opt.label == -1 else opt.label)+'_'+str(opt.gen_qtd)+('_trunc' + str(opt.trunc) if opt.trunc_mode!='-' else '')+('_stochastic' if opt.stochastic else '')+'_gen_label.pkl'), 'wb') as f:
    pickle.dump(new_labels, f)
