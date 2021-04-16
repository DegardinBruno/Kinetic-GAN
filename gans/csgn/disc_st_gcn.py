import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.tgcn import ConvTemporalGraphical
from utils.graph import Graph


class Discriminator(nn.Module):
    
    def __init__(self, in_channels, edge_importance_weighting=True, **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph()
        self.A = [torch.tensor(Al, dtype=torch.float32, requires_grad=False).cuda() for Al in self.graph.As]

        # build networks
        spatial_kernel_size  = [A.size(0) for A in self.A]
        temporal_kernel_size = [9-2*(i) for i, A in enumerate(self.A)]
        kernel_size          = (temporal_kernel_size, spatial_kernel_size)
        t_size               = 300
        self.data_bn = nn.BatchNorm1d(in_channels * self.A[0].size(1))

        #kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 32, kernel_size, 1, lvl=0, up_s=False, up_t=t_size, residual=False, **kwargs),
            st_gcn(32, 64, kernel_size, 1, lvl=1, up_s=True, up_t=t_size, **kwargs),
            st_gcn(64, 128, kernel_size, 1, lvl=1, up_s=False, up_t=int(t_size/2), **kwargs),
            st_gcn(128, 256, kernel_size, 1, lvl=2, up_s=True, up_t=int(t_size/4), **kwargs),
            st_gcn(256, 512, kernel_size, 1, lvl=2, up_s=False, up_t=int(t_size/8),  **kwargs),
            st_gcn(512, 1024, kernel_size, 1, lvl=3, tan=True, up_s=True, up_t=int(t_size/16),  **kwargs),
        ))


        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A[i.lvl].size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(1024, 2, kernel_size=1)

    def forward(self, x):


        # data normalization
        N, C, T, V = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(N, C, T, V)
        
        print(x.shape)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A[gcn.lvl] * importance)

        
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, -1, 1, 1)

        print(x.shape)
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)

        print(x.shape)
        return x

    

class st_gcn(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                lvl=3,
                dropout=0,
                residual=True,
                up_s=False, up_t=300, tan=False):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0][lvl] % 2 == 1
        padding = ((kernel_size[0][lvl] - 1) // 2, 0)
        self.lvl, self.up_s, self.up_t, self.tan = lvl, up_s, up_t, tan
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                        kernel_size[1][lvl])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0][lvl], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
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
                nn.BatchNorm2d(out_channels),
            )

        self.l_relu = nn.LeakyReLU(0.2, inplace=True)
        self.tanh   = nn.Tanh()

    def forward(self, x, A):
        
        if self.up_s:
            x = F.interpolate(x, size=(x.size(2), A.size(1)))  # Exactly like nn.Upsample

        x = F.interpolate(x, size=(self.up_t,x.size(-1)))  # Exactly like nn.Upsample


        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res


        return self.tanh(x) if self.tan else self.l_relu(x), A