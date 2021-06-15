import numpy as np
import os, sys
import argparse
import matplotlib.pyplot as plt


sys.path.append(".")

from utils import general

parser = argparse.ArgumentParser()
parser.add_argument("--batches", type=int, default=1253, help="Batches per epoch")
parser.add_argument("--runs", type=str, default="kinetic-gan", help="Runs file")
parser.add_argument("--exp", type=int, default=-1, help="Experiment ID")
opt = parser.parse_args()
print(opt)

out  = general.check_runs(opt.runs, id=opt.exp)
test = general.load(opt.runs, 'plot_loss', run_id=opt.exp)

d_loss = np.concatenate(test['d_loss'])
g_loss = np.concatenate(test['g_loss'])

d_loss = d_loss[:int(len(d_loss)/opt.batches)*opt.batches]
g_loss = g_loss[:int(len(g_loss)/opt.batches)*opt.batches]
d_loss = np.array(np.split(d_loss, int(len(d_loss)/opt.batches)))
g_loss = np.array(np.split(g_loss, int(len(g_loss)/opt.batches)))


d_loss = [np.mean(loss) for loss in d_loss]
g_loss = [np.mean(loss) for loss in g_loss]


x_iter = np.arange(0,len(d_loss),1)


plt.clf()
plt.plot(x_iter, d_loss, color='blue', linewidth=1, label='D loss', alpha=0.6)
plt.plot(x_iter, g_loss, color='red', linewidth=1, label='G loss', alpha=0.6)


plt.title('Kinetic-GAN Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(out, 'loss.pdf'))
plt.show()
