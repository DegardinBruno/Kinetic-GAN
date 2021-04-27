import numpy as np, os, re, sys
from feeder.cgan_feeder import Feeder
from collections import Counter

import torch
import torch.nn as nn

labels = np.array([num for _ in range(10) for num in range(60)])

print(labels)
