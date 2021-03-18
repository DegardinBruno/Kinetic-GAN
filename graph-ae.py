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
from utils.general import check_runs

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='NTU-RGB+D', help='type of dataset.')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
parser.add_argument('--train_path', type=str, default='/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_data.npy', help='type of dataset.')
parser.add_argument('--train_label_path', type=str, default='/media/socialab/bb715954-b8c5-414e-b2e1-95f4d2ff6f3d/ST-GCN/NTU-RGB-D/xview/train_label.pkl', help='type of dataset.')

args = parser.parse_args()

out = check_runs('graph-ae')
model_out = os.path.join(out, 'models')
os.makedirs(model_out)


def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))


    device = torch.device('cuda:0')

    # Configure dataloaders
    dataset = Feeder(args.train_path, args.train_label_path)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )

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
    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray()).to(device)

    pos_weight = np.asarray(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    model = GCNModelVAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    features = torch.FloatTensor(torch.from_numpy(dataset[100])).to(device)

    hidden_emb = None
    model.train()
    for epoch in range(args.epochs):
        t = time.time()

        #loss_ = []
        #ap_   = []
        #for i, features in enumerate(dataloader):

        recovered, mu, logvar = model(features, adj_norm)
        loss = loss_function(preds=recovered, labels=adj_label,
                            mu=mu, logvar=logvar, n_nodes=n_nodes,
                            norm=norm, pos_weight=torch.from_numpy(pos_weight))

        optimizer.zero_grad()
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        hidden_emb = mu.data.cpu().numpy()
        roc_curr, ap_curr = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false, False)

        #loss_.append(cur_loss)
        #ap_.append(ap_curr)

        #if i>2000:
        #    break

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
                "val_ap=", "{:.5f}".format(ap_curr),
                "time=", "{:.5f}".format(time.time() - t)
                )

    
        torch.save(model.state_dict(), os.path.join(model_out,'graph-ae_'+str(epoch)+'.pt'))

    print("Optimization Finished!")

    roc_score, ap_score = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false, True)
    print('Test ROC score: ' + str(roc_score))
    print('Test AP score: ' + str(ap_score))


if __name__ == '__main__':
    gae_for(args)