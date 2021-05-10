import argparse
import os
import numpy as np
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F

from utils.c_gen_st_gcn import Generator
#from utils.c_disc_st_gcn import Discriminator
# from utils.st_gcn import Discriminator
from feeder.cgan_feeder import Feeder
from utils import general


out        = general.check_runs('cgc-gan')
models_out  = os.path.join(out, 'models')
images_out = os.path.join(out, 'images')
if not os.path.exists(models_out): os.makedirs(models_out)
if not os.path.exists(images_out): os.makedirs(images_out)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=1024, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=60, help="number of classes for dataset")
parser.add_argument("--t_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--v_size", type=int, default=25, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--checkpoint_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--d_interval", type=int, default=1, help="interval of interation for discriminator")
parser.add_argument("--data_path", type=str, default="/home/degar/DATASETS/st-gcn/NTU/xview/train_data.npy", help="path to data")
parser.add_argument("--label_path", type=str, default="/home/degar/DATASETS/st-gcn/NTU/xview/train_label.pkl", help="path to label")
opt = parser.parse_args()
print(opt)

config_file = open(os.path.join(out,"config.txt"),"w")
config_file.write(str(os.path.basename(__file__)) + '|' + str(opt))
config_file.close()

img_shape = (opt.channels, opt.t_size, opt.v_size)


cuda = True if torch.cuda.is_available() else False
print(cuda)



# Loss weight for gradient penalty
lambda_gp = 10


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + opt.n_classes, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.shape[0], -1), self.label_emb(labels)), -1)
        validity = self.model(d_in)
        return validity

# Loss functions
# adversarial_loss = torch.nn.BCELoss()  # For normal sampling
# adversarial_loss = torch.nn.MSELoss()  # For conditional sampling
# auxiliary_loss   = torch.nn.NLLLoss()    # For conditional sampling (Aux-Class GAN)

generator     = Generator(opt.latent_dim, opt.n_classes)
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    # auxiliary_loss.cuda()


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


def sample_image(n_row, batches_done):
    z = Variable(Tensor(np.random.normal(0, 1, (10*n_row, opt.latent_dim)))) # , int(opt.t_size/16), 1
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(10) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    with open(os.path.join(images_out, str(batches_done)+'.npy'), 'wb') as npf:
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

        imgs = imgs[:,:,:opt.t_size,:]
        batches_done = epoch * len(dataloader) + i

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        labels    = Variable(labels.type(LongTensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))  # ATTENTION , int(opt.t_size/16), 1

        # Generate a batch of images
        fake_imgs = generator(z, labels)

        # Real images
        real_validity = discriminator(real_imgs, labels)
        # Fake images
        fake_validity = discriminator(fake_imgs, labels)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, labels.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z, labels)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
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
            sample_image(n_row=60, batches_done=batches_done)

            general.save('cgc-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')
        
        if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), os.path.join(models_out, "generator_%d.pth" % batches_done))
            torch.save(discriminator.state_dict(), os.path.join(models_out, "discriminator_%d.pth" % batches_done))

loss_d = np.array(loss_d)
loss_g = np.array(loss_g)

general.save('cgc-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')



'''

# ----------
#  Training
# ----------


loss_d = []
loss_g = []

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i

        imgs = imgs[:,:,:opt.t_size,:]
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim, 1, 1))))  # ATTENTION int(opt.t_size/16)
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if batches_done % opt.d_interval == 0:
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach(), gen_labels)
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        if batches_done % 100 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

        loss_d.append(d_loss.data.cpu())
        loss_g.append(g_loss.data.cpu())

        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=60, batches_done=batches_done)

            general.save('cgc-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')
        
        if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), os.path.join(models_out, "generator_%d.pth" % batches_done))
            torch.save(discriminator.state_dict(), os.path.join(models_out, "discriminator_%d.pth" % batches_done))

loss_d = np.array(loss_d)
loss_g = np.array(loss_g)

general.save('cgc-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')

'''

