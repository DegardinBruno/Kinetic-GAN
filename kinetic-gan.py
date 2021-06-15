import argparse
import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from shutil import copyfile

from utils.generator import Generator
from utils.discriminator import Discriminator
from feeder.feeder import Feeder
from utils import general



out        = general.check_runs('kinetic-gan')
models_out  = os.path.join(out, 'models')
actions_out = os.path.join(out, 'actions')
if not os.path.exists(models_out): os.makedirs(models_out)
if not os.path.exists(actions_out): os.makedirs(actions_out)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=512, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=60, help="number of classes for dataset")
parser.add_argument("--t_size", type=int, default=64, help="size of each temporal dimension")
parser.add_argument("--v_size", type=int, default=25, help="size of each spatial dimension (vertices)")
parser.add_argument("--channels", type=int, default=3, help="number of channels (coordinates)")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per generator's iteration")
parser.add_argument("--lambda_gp", type=int, default=10, help="Loss weight for gradient penalty in WGAN-GP Loss")
parser.add_argument("--sample_interval", type=int, default=5000, help="interval between action sampling")
parser.add_argument("--checkpoint_interval", type=int, default=10000, help="interval between model saving")
parser.add_argument("--data_path", type=str, default="/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU/xsub/train_data.npy", help="path to data")
parser.add_argument("--label_path", type=str, default="/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU/xsub/train_label.pkl", help="path to label")
opt = parser.parse_args()
print(opt)

# Save config file and respective generator and discriminator for reproducibilty
config_file = open(os.path.join(out,"config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
config_file.close()

copyfile(os.path.basename(__file__), os.path.join(out, os.path.basename(__file__)))
copyfile('utils/generator.py', os.path.join(out, 'generator.py'))
copyfile('utils/discriminator.py', os.path.join(out, 'discriminator.py'))

cuda = True if torch.cuda.is_available() else False
print('CUDA',cuda)

# Models initialization
generator     = Generator(opt.latent_dim, opt.n_classes, opt.t_size)
discriminator = Discriminator(opt.channels, opt.n_classes, opt.t_size, opt.latent_dim)

if cuda:
    generator.cuda()
    discriminator.cuda()


# Configure data loader
dataloader = torch.utils.data.DataLoader(
    dataset=Feeder(opt.data_path, opt.label_path),
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=opt.n_cpu
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor     = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_action(n_row, batches_done):
    z = Variable(Tensor(np.random.normal(0, 1, (10*n_row, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(10) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    with open(os.path.join(actions_out, str(batches_done)+'.npy'), 'wb') as npf:
        np.save(npf, gen_imgs.data.cpu())


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------

loss_d, loss_g = [], []
batches_done   = 0
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batches_done = epoch * len(dataloader) + i

        # Configure input
        imgs = imgs[:,:,:opt.t_size,:]
        real_imgs = Variable(imgs.type(Tensor))
        labels    = Variable(labels.type(LongTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        # Generate a batch of actions
        fake_imgs = generator(z, labels)

        # Real actions
        real_validity = discriminator(real_imgs, labels)
        # Fake actions
        fake_validity = discriminator(fake_imgs, labels)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + opt.lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator after n_critic discriminator steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of actions
            fake_imgs = generator(z, labels)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake actions
            fake_validity = discriminator(fake_imgs, labels)
            g_loss = -torch.mean(fake_validity)

            g_loss.backward()
            optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        loss_d.append(d_loss.data.cpu())
        loss_g.append(g_loss.data.cpu())

        if batches_done % opt.sample_interval == 0:
            sample_action(n_row=opt.n_classes, batches_done=batches_done)

            general.save('kinetic-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')
        
        if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), os.path.join(models_out, "generator_%d.pth" % batches_done))
            torch.save(discriminator.state_dict(), os.path.join(models_out, "discriminator_%d.pth" % batches_done))

loss_d = np.array(loss_d)
loss_g = np.array(loss_g)

general.save('kinetic-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')