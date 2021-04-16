import argparse
import os
import numpy as np
import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from gen_st_gcn import Generator
from disc_st_gcn import Discriminator
cuda = True if torch.cuda.is_available() else False
print(cuda)


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

T = 300


generator     = Generator(1024)
discriminator = Discriminator(3)

if cuda:
    generator.cuda()
    discriminator.cuda()

z = Variable(FloatTensor(np.random.normal(0, 1, (16, 1024, int(T/16), 1))))


fake_img = generator(z)
ans      = discriminator(fake_img)

