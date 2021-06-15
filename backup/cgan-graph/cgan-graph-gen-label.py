import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from feeder.cgan_feeder import Feeder
from utils import general

out        = general.check_runs('cgan-graph')
models_out  = os.path.join(out, 'models')
images_out = os.path.join(out, 'images')
if not os.path.exists(models_out): os.makedirs(models_out)
if not os.path.exists(images_out): os.makedirs(images_out)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=60, help="number of classes for dataset")
parser.add_argument("--t_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--img_size", type=int, default=25, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=10000, help="interval between image sampling")
parser.add_argument("--checkpoint_interval", type=int, default=10000, help="interval between image sampling")
parser.add_argument("--d_interval", type=int, default=1, help="interval of interation for discriminator")
parser.add_argument("--data_path", type=str, default="/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_data.npy", help="path to data")
parser.add_argument("--label_path", type=str, default="/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_label.pkl", help="path to label")
parser.add_argument("--gen_model_path", type=str, default="runs/cgan-graph/shift-ntu-epoch1000/models/generator_2350000.pth", help="path to gen model")
parser.add_argument("--disc_model_path", type=str, default="runs/cgan-graph/shift-ntu-epoch1000/models/discriminator_2350000.pth", help="path to gen model")
parser.add_argument("--gen_label", type=int, default=59, help="Action to generate, default: Jump")
parser.add_argument("--gen_qtd", type=int, default=1000, help="How many samples to generate per class")
parser.add_argument("--threshold", type=float, default=0.90, help="Confidence threshold")
opt = parser.parse_args()
print(opt)

config_file = open(os.path.join(out,"config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
config_file.close()

img_shape = (opt.channels, opt.t_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
print(cuda)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# Load Models
generator.load_state_dict(torch.load(opt.gen_model_path))
generator.eval()
discriminator.load_state_dict(torch.load(opt.disc_model_path))
discriminator.eval()

# Generate Samples and Labels
z = Variable(FloatTensor(np.random.normal(0, 1, (opt.gen_qtd, opt.latent_dim))))
labels_np = np.array([opt.gen_label for _ in range(opt.gen_qtd)])
labels = Variable(LongTensor(labels_np))
gen_imgs = generator(z, labels)

print(gen_imgs.shape)

# Samples validity with discriminator
validity = discriminator(gen_imgs, labels).data.cpu().numpy()
filter_val = np.where(validity>opt.threshold)[0]

print(filter_val)
print(filter_val.shape)

gen_imgs = gen_imgs[filter_val]
labels_np = labels_np[filter_val]

print(gen_imgs.shape)
print(labels_np.shape)

if len(filter_val)>0:
    with open(os.path.join(images_out, str(opt.gen_label)+'_'+str(opt.gen_qtd)+'_gen_data.npy'), 'wb') as npf:
        np.save(npf, gen_imgs.data.cpu())

    with open(os.path.join(images_out, str(opt.gen_label)+'_'+str(opt.gen_qtd)+'_gen_label.npy'), 'wb') as npf:
        np.save(npf, labels_np)

















