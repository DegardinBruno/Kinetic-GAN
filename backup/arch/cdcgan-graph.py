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

out        = general.check_runs('cdcgan-graph')
models_out  = os.path.join(out, 'models')
images_out = os.path.join(out, 'images')
if not os.path.exists(models_out): os.makedirs(models_out)
if not os.path.exists(images_out): os.makedirs(images_out)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=60, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='interval between image sampling')
parser.add_argument("--data_path", type=str, default="/home/degar/DATASETS/st-gcn/NTU/xview/train_data.npy", help="path to data")
parser.add_argument("--label_path", type=str, default="/home/degar/DATASETS/st-gcn/NTU/xview/train_label.pkl", help="path to label")
args = parser.parse_args()
print(args)

C,H,W = args.channels, args.img_size, args.img_size


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(args.n_classes, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, C, 4, 2, 1)


    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(C, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(args.n_classes, d//2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)


    # def forward(self, input):
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))
        return x

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize Generator and discriminator
generator = Generator()
discriminator = Discriminator()

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
# Configure data loader
dataloader = torch.utils.data.DataLoader(
    dataset=Feeder(args.data_path, args.label_path),
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=4
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))



def sample_image(N_Class, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (N_Class**2, args.latent_dim,1,1))).cuda())
    #fixed labels
    y_ = torch.LongTensor(np.array([num for num in range(N_Class)])).view(N_Class,1).expand(-1,N_Class).contiguous()
    y_fixed = torch.zeros(N_Class**2, N_Class)
    y_fixed = Variable(y_fixed.scatter_(1,y_.view(N_Class**2,1),1).view(N_Class**2, N_Class,1,1).cuda())

    gen_imgs = generator(noise, y_fixed).view(-1,C,H,W)
    
    with open(os.path.join(images_out, str(batches_done)+'.npy'), 'wb') as npf:
        np.save(npf, gen_imgs.data.cpu())

loss_d = []
loss_g = []

zpad = nn.ZeroPad2d((4,3,0,0))

for epoch in range(args.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        imgs = imgs[:,:,:32,:]


        Batch_Size = args.batch_size
        N_Class = args.n_classes
        img_size = args.img_size
        # Adversarial ground truths
        valid = Variable(torch.ones(Batch_Size).cuda(), requires_grad=False)
        fake = Variable(torch.zeros(Batch_Size).cuda(), requires_grad=False)

        # Configure input
        real_imgs = Variable(zpad(imgs.type(torch.FloatTensor)).cuda())

        real_y = torch.zeros(Batch_Size, N_Class)
        real_y = real_y.scatter_(1, labels.view(Batch_Size, 1), 1).view(Batch_Size, N_Class, 1, 1).contiguous()
        real_y = Variable(real_y.expand(-1, -1, img_size, img_size).cuda())

        # Sample noise and labels as generator input
        noise = Variable(torch.randn((Batch_Size, args.latent_dim,1,1)).cuda())
        gen_labels = (torch.rand(Batch_Size, 1) * N_Class).type(torch.LongTensor)
        gen_y = torch.zeros(Batch_Size, N_Class)
        gen_y = Variable(gen_y.scatter_(1, gen_labels.view(Batch_Size, 1), 1).view(Batch_Size, N_Class,1,1).cuda())

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Loss for real images
        d_real_loss = adversarial_loss(discriminator(real_imgs, real_y).squeeze(), valid)
        # Loss for fake images
        gen_imgs = generator(noise, gen_y)
        gen_y_for_D = gen_y.view(Batch_Size, N_Class, 1, 1).contiguous().expand(-1, -1, img_size, img_size)

        d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y_for_D).squeeze(), fake)
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss)
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        g_loss = adversarial_loss(discriminator(gen_imgs,gen_y_for_D).squeeze(), valid)
        g_loss.backward()
        optimizer_G.step()



        loss_d.append(d_loss.data.cpu())
        loss_g.append(g_loss.data.cpu())


        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs, i, len(dataloader),
                                                            d_loss.data.cpu(), g_loss.data.cpu()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % args.sample_interval == 0:
            sample_image(N_Class, batches_done)
            
            general.save('cdcgan-graph', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')


loss_d = np.array(loss_d)
loss_g = np.array(loss_g)

general.save('cdcgan-graph', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')