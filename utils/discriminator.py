import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .init_gan.tgcn import ConvTemporalGraphical
from .init_gan.graph_ntu import graph_ntu
from .init_gan.graph_h36m import Graph_h36m
import numpy as np


class Discriminator(nn.Module):
    
    def __init__(self, in_channels, n_classes, t_size, latent, edge_importance_weighting=True, dataset='ntu', **kwargs):
        super().__init__()

        # load graph
        self.graph = graph_ntu() if dataset == 'ntu' else Graph_h36m()
        self.A = [torch.tensor(Al, dtype=torch.float32, requires_grad=False).cuda() for Al in self.graph.As]

        # build networks
        spatial_kernel_size  = [A.size(0) for A in self.A]
        temporal_kernel_size = [3 for _ in self.A]
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)
        self.t_size          = t_size

        #kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels+n_classes, 32, kernel_size, 1, graph=self.graph, lvl=0, dw_s=True, dw_t=t_size, residual=False, **kwargs),
            st_gcn(32, 64, kernel_size, 1, graph=self.graph, lvl=1, dw_s=False, dw_t=t_size, **kwargs),
            st_gcn(64, 128, kernel_size, 1, graph=self.graph, lvl=1, dw_s=True, dw_t=int(t_size/2), **kwargs),
            st_gcn(128, 256, kernel_size, 1, graph=self.graph, lvl=2, dw_s=False, dw_t=int(t_size/4), **kwargs),
            st_gcn(256, 512, kernel_size, 1, graph=self.graph, lvl=2, dw_s=True, dw_t=int(t_size/8),  **kwargs),
            st_gcn(512, latent, kernel_size, 1, graph=self.graph, lvl=3, tan=False, dw_s=False, dw_t=int(t_size/16),  **kwargs),
        ))


        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A[i.lvl].size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.label_emb = nn.Embedding(n_classes, n_classes)

        # fcn for prediction
        self.fcn = nn.Linear(latent, 1)

    def forward(self, x, labels):
        
        N, C, T, V = x.size()


        c = self.label_emb(labels)
        c = c.view(c.size(0), c.size(1), 1, 1).repeat(1, 1, T, V)

        x = torch.cat((c, x), 1)
        
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A[gcn.lvl] * importance)

        
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1)

        # prediction
        validity = self.fcn(x)

        return validity

    

class st_gcn(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                graph=None,
                lvl=3,
                dropout=0,
                residual=True,
                dw_s=False, dw_t=64, tan=False):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0][lvl] % 2 == 1
        padding = ((kernel_size[0][lvl] - 1) // 2, 0)
        self.graph, self.lvl, self.dw_s, self.dw_t, self.tan = graph, lvl, dw_s, dw_t, tan
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                        kernel_size[1][lvl])

        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0][lvl], 1),
                (stride, 1),
                padding,
            ),
        )


        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
            )


        self.l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh   = nn.Tanh()

    def forward(self, x, A):
        

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x    = self.tcn(x) + res

        x = self.downsample_s(x) if self.dw_s else x
        
        x = F.interpolate(x, size=(self.dw_t,x.size(-1)))  # Exactly like nn.Upsample

        return self.tanh(x) if self.tan else self.l_relu(x), A


    def downsample_s(self, tensor):
        keep = self.graph.map[self.lvl+1][:,1]

        return tensor[:,:,:,keep]