import numpy as np
import os, re, random
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt

from utils.general import *


test = load('graph-ae', 'gae')

print(test['fpr'])