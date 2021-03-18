from __future__ import division
from __future__ import print_function

import argparse
import time, os

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.utils.data import DataLoader

from utils.gae_model import GCNModelVAE
from utils.gae_optimizer import loss_function
from utils.gae_utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from feeder.gae_feeder import Feeder

import matplotlib.pyplot as plt
import networkx as nx
from utils.general import check_runs

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='NTU-RGB+D', help='type of dataset.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--train_path', type=str, default='/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_data.npy', help='type of dataset.')
parser.add_argument('--train_label_path', type=str, default='/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_label.pkl', help='type of dataset.')

args = parser.parse_args()

out = check_runs('graph-ae', -1)

np.random.seed(args.seed)

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))


    device = torch.device('cuda:0')

    # Configure dataset
    dataset = Feeder(args.train_path, args.train_label_path)

    adj = load_data()
    n_nodes, feat_dim = dataset.V, dataset.C

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train


    # Some preprocessing
    adj_norm = preprocess_graph(adj).to(device)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    checkpoint = torch.load(os.path.join(out, 'models/graph-ae_999.pt'))
    model.load_state_dict(checkpoint)
    model.to(device)

    hidden_emb = None
    model.eval()

    input_f = torch.FloatTensor(dataset[100]).to(device)

    print(input_f.shape)

    recovered, mu, logvar = model(input_f, adj_norm)


    hidden_emb = mu.data.cpu().numpy()

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false, saving=True)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))

    #print(adj_rec)

    '''adj_rec = sigmoid(np.dot(hidden_emb, hidden_emb.T))
    adj_rec[np.where(adj_rec>0.99)] = 1
    adj_rec[np.where(adj_rec<0.99)] = 0'''

    #print(adj_rec)

    rows, cols = np.where(adj_rec == 1.0)
    #print(rows)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    plt.clf()
    nx.draw_networkx(gr)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
    gae_for(args)