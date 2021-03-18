import numpy as np
import os, re, random
import pickle
import torch
import networkx as nx
import matplotlib.pyplot as plt

def humanSort(text):  # Sort function for strings w/ numbers
    convText = lambda seq: int(seq) if seq.isdigit() else seq.lower()
    arrayKey = lambda key: [convText(s) for s in re.split('([0-9]+)', key)]  # Split numbers and chars, base function for sorted
    return sorted(text, key=arrayKey)


num_node = 25
self_link = [(i, i) for i in range(num_node)]
neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                    (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                    (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                    (22, 23), (23, 8), (24, 25), (25, 12)]
neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
edge = self_link + neighbor_link

G = nx.Graph(edge)
A = nx.adjacency_matrix(G)
print(A.shape)
nx.draw_networkx(G)

plt.show()
