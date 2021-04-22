import argparse
import os
import numpy as np
import math
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F

from utils.gen_st_gcn import Generator
from utils.disc_st_gcn import Discriminator
from feeder.cgan_feeder import Feeder
from utils import general


out        = general.check_runs('gc-gan')
models_out  = os.path.join(out, 'models')
images_out = os.path.join(out, 'images')
if not os.path.exists(models_out): os.makedirs(models_out)
if not os.path.exists(images_out): os.makedirs(images_out)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=1024, help="dimensionality of the latent space")
#parser.add_argument("--n_classes", type=int, default=60, help="number of classes for dataset")
parser.add_argument("--t_size", type=int, default=300, help="size of each image dimension")
#parser.add_argument("--img_size", type=int, default=25, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
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



cuda = True if torch.cuda.is_available() else False
print(cuda)

# Loss weight for gradient penalty
lambda_gp = 10


# Loss functions
# adversarial_loss = torch.nn.BCELoss()

generator     = Generator(opt.latent_dim)
discriminator = Discriminator(opt.channels)

if cuda:
    generator.cuda()
    discriminator.cuda()
    #adversarial_loss.cuda()


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

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_image(batches_done):
    z = Variable(Tensor(np.random.normal(0, 1, (100, opt.latent_dim, opt.t_size, 1))))
    gen_imgs = generator(z)
    with open(os.path.join(images_out, str(batches_done)+'.npy'), 'wb') as npf:
        np.save(npf, gen_imgs.data.cpu())


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
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
    for i, (imgs, _) in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim, int(opt.t_size/16), 1))))  # ATTENTION

        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
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
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
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
            sample_image(batches_done=batches_done)

            general.save('gc-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')
        
        if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), os.path.join(models_out, "generator_%d.pth" % batches_done))
            torch.save(discriminator.state_dict(), os.path.join(models_out, "discriminator_%d.pth" % batches_done))

loss_d = np.array(loss_d)
loss_g = np.array(loss_g)

general.save('cgan-graph', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')

'''# ----------
#  Training
# ----------

loss_d, loss_g = [], []
batches_done   = 0

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i



        # Adversarial ground truths
        valid = Variable(Tensor(opt.batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(opt.batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim, int(opt.t_size/16), 1))))  # ATTENTION

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        if batches_done % opt.d_interval == 0:
            optimizer_D.zero_grad()

            # Loss for real images
            validity_real = discriminator(real_imgs)
            d_real_loss = adversarial_loss(validity_real, valid)

            # Loss for fake images
            validity_fake = discriminator(gen_imgs.detach())
            d_fake_loss = adversarial_loss(validity_fake, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        loss_d.append(d_loss.data.cpu())
        loss_g.append(g_loss.data.cpu())

        if batches_done % opt.sample_interval == 0:
            sample_image(batches_done=batches_done)

            general.save('gc-gan', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')
        
        if opt.checkpoint_interval != -1 and batches_done % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), os.path.join(models_out, "generator_%d.pth" % batches_done))
            torch.save(discriminator.state_dict(), os.path.join(models_out, "discriminator_%d.pth" % batches_done))

loss_d = np.array(loss_d)
loss_g = np.array(loss_g)

general.save('cgan-graph', {'d_loss': loss_d, 'g_loss': loss_g}, 'plot_loss')'''

