import time
import torch
import torch.nn as nn 
import torch.nn.functional as F

from networks.basenet import BaseNet
from torch_geometric.utils import scatter_
from torch_geometric.nn import knn_graph, EdgeConv
from torch_geometric.nn.inits import reset
          
class MLP(nn.Module):
    def __init__(self, layer_dims, batch_norm=True, relu=True):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i+1]
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if relu:
                layers.append(nn.ReLU()) 
        self.layer = nn.Sequential(*layers)  
    
    def reset_parameters(self):
        reset(self.layer)
        
    def forward(self, x):
        return self.layer(x)
     
class DGCNNCls(BaseNet):
    def __init__(self, num_classes, K=10, conv_aggr = 'max', device=None):
        super().__init__()
        print('Build up DGCNNCls model...')
        self.K = K
        self.num_classes = num_classes
        self.conv_aggr = conv_aggr
        self.edge_conv_dims = [3, 64, 64, 64, 128]
        self.edge_convs = self.make_edge_conv_layers_()
        self.glb_aggr = MLP([sum(self.edge_conv_dims[1::]), 1024])
        self.fc = nn.Sequential(MLP([1024, 512]), nn.Dropout(0.5),
                                MLP([512, 256]) , nn.Dropout(0.5),
                                nn.Linear(256, self.num_classes))
        self.device = device
        self.to(self.device)
        
    def make_edge_conv_layers_(self):
        """Define structure of the EdgeConv Blocks
        edge_conv_dims: [[convi_mlp_dims]], e.g., [[3, 64], [64, 128]]
        """
        layers = []
        dims =  self.edge_conv_dims
        for i in range(len(dims) - 1):
            mlp_dims = [dims[i] * 2, dims[i+1]]
            layers.append(EdgeConv(nn=MLP(mlp_dims), aggr=self.conv_aggr))
        return nn.Sequential(*layers)
    
    def init_weights_(self):
        print('Initialize all model parameters: TBD')
        pass
                    
    def forward(self, pts, batch_ids):
        """
        Input: 
            - data.pos: (B*N, 3) 
            - data.batch: (B*N,)
        Return:
            - out: (B, C), softmax prob
        """
        batch_size = max(batch_ids) + 1  
        out = pts
        edge_conv_outs = []
        for edge_conv in self.edge_convs.children():
            # Dynamically update graph
            edge_index = knn_graph(pts, k=self.K, batch=batch_ids)
            out = edge_conv(out, edge_index)
            edge_conv_outs.append(out) 
        out = torch.cat(edge_conv_outs, dim=-1)  # Skip connection to previous features
        out = self.glb_aggr(out)  # Global aggregation
        out = scatter_('max', out, index=batch_ids, dim_size=batch_size)
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
    
    def loss_(self, pts, batch_ids, lbls):
        logits = self.forward(pts, batch_ids)
        loss = F.nll_loss(logits, lbls)
        return loss
    
    def pred_(self, pts, batch_ids):
        return self.forward(pts, batch_ids).max(1).indices
    