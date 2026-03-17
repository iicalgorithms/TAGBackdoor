"""Graph Neural Network models for TAG backdoor attack experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_mean_pool, GINConv, ChebConv, APPNP, TransformerConv
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_scatter import scatter_add
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Optional, Dict, Any
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from copy import deepcopy
import logging

class GCN(nn.Module):
    """Graph Convolutional Network."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 add_self_loops: bool = True):
        super(GCN, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim, add_self_loops=add_self_loops))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=add_self_loops))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim, add_self_loops=add_self_loops))
        else:
            self.convs[0] = GCNConv(input_dim, output_dim)
    
    def forward(self, x, edge_index, return_embeddings=False, edge_weight=None):
        # Apply convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        embeddings = x
        x = self.convs[-1](x, edge_index, edge_weight=edge_weight)
        
        if return_embeddings:
            return x, embeddings
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE model."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 aggr: str = 'mean'):
        super(GraphSAGE, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggr))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim, aggr=aggr))
        else:
            self.convs[0] = SAGEConv(input_dim, output_dim, aggr=aggr)
    
    def forward(self, x, edge_index, return_embeddings=False):
        # Apply convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        embeddings = x
        x = self.convs[-1](x, edge_index)
        
        if return_embeddings:
            return x, embeddings
        return x

class GAT(nn.Module):
    """Graph Attention Network."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 heads: int = 8,
                 concat: bool = False):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, concat=concat, dropout=dropout))
        
        # Adjust dimensions based on concatenation
        layer_input_dim = hidden_dim * heads if concat else hidden_dim
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(layer_input_dim, hidden_dim, heads=heads, concat=concat, dropout=dropout))
        
        # Output layer (no concatenation for final layer)
        if num_layers > 1:
            self.convs.append(GATConv(layer_input_dim, output_dim, heads=1, concat=False, dropout=dropout))
        else:
            self.convs[0] = GATConv(input_dim, output_dim, heads=1, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index, return_embeddings=False):
        # Apply convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final layer
        embeddings = x
        x = self.convs[-1](x, edge_index)
        
        if return_embeddings:
            return x, embeddings
        return x


class GIN(nn.Module):
    """Graph Isomorphism Network."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 eps: float = 0.0,
                 train_eps: bool = False):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Build MLPs for each layer
        self.convs = nn.ModuleList()

        # Input layer
        mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))

        # Hidden layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))

        # Output layer
        if num_layers > 1:
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.convs.append(GINConv(mlp, eps=eps, train_eps=train_eps))
        else:
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
            self.convs[0] = GINConv(mlp, eps=eps, train_eps=train_eps)

    def forward(self, x, edge_index, return_embeddings=False):
        # Apply convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        embeddings = x
        x = self.convs[-1](x, edge_index)

        if return_embeddings:
            return x, embeddings
        return x


class ChebNet(nn.Module):
    """Chebyshev Spectral Graph Convolutional Network."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 K: int = 3):
        super(ChebNet, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(input_dim, hidden_dim, K=K))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(ChebConv(hidden_dim, hidden_dim, K=K))

        # Output layer
        if num_layers > 1:
            self.convs.append(ChebConv(hidden_dim, output_dim, K=K))
        else:
            self.convs[0] = ChebConv(input_dim, output_dim, K=K)

    def forward(self, x, edge_index, return_embeddings=False):
        # Apply convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        embeddings = x
        x = self.convs[-1](x, edge_index)

        if return_embeddings:
            return x, embeddings
        return x


class APPNPNet(nn.Module):
    """Approximate Personalized Propagation of Neural Predictions."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 K: int = 10,
                 alpha: float = 0.1):
        super(APPNPNet, self).__init__()

        self.dropout = dropout

        # MLP layers
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(input_dim, hidden_dim))

        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))

        if num_layers > 1:
            self.lins.append(nn.Linear(hidden_dim, output_dim))
        else:
            self.lins[0] = nn.Linear(input_dim, output_dim)

        # APPNP propagation
        self.prop = APPNP(K=K, alpha=alpha)

    def forward(self, x, edge_index, return_embeddings=False):
        # Apply MLP layers
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Store embeddings before final layer
        embeddings = x

        # Final MLP layer
        x = self.lins[-1](x)

        # APPNP propagation
        x = self.prop(x, edge_index)

        if return_embeddings:
            return x, embeddings
        return x


class GraphTransformer(nn.Module):
    """Graph Transformer Network."""

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 heads: int = 8,
                 concat: bool = False,
                 beta: bool = False,
                 edge_dim: Optional[int] = None):
        super(GraphTransformer, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Input layer
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(
            input_dim, hidden_dim, heads=heads, concat=concat,
            beta=beta, dropout=dropout, edge_dim=edge_dim
        ))

        # Adjust dimensions based on concatenation
        layer_input_dim = hidden_dim * heads if concat else hidden_dim

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(
                layer_input_dim, hidden_dim, heads=heads, concat=concat,
                beta=beta, dropout=dropout, edge_dim=edge_dim
            ))

        # Output layer (no concatenation for final layer)
        if num_layers > 1:
            self.convs.append(TransformerConv(
                layer_input_dim, output_dim, heads=1, concat=False,
                beta=beta, dropout=dropout, edge_dim=edge_dim
            ))
        else:
            self.convs[0] = TransformerConv(
                input_dim, output_dim, heads=1, concat=False,
                beta=beta, dropout=dropout, edge_dim=edge_dim
            )

    def forward(self, x, edge_index, return_embeddings=False, edge_attr=None):
        # Apply convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final layer
        embeddings = x
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)

        if return_embeddings:
            return x, embeddings
        return x


class MLP(nn.Module):
    """Multi-Layer Perceptron (no graph structure)."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super(MLP, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        if num_layers > 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers = [nn.Linear(input_dim, output_dim)]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x, edge_index=None, return_embeddings=False):
        # MLP doesn't use edge_index
        if return_embeddings:
            # Get embeddings from second-to-last layer
            embeddings = x
            for layer in self.layers[:-1]:
                embeddings = layer(embeddings)
            
            output = self.layers[-1](embeddings)
            return output, embeddings
        
        return self.layers(x)

def accuracy(output, labels):
    if not hasattr(labels, '__len__'):  # 检查标签是否具有长度属性（例如，单个标签的情况）
        labels = [labels]  # 如果没有，将其转换为列表
    if type(labels) is not torch.Tensor:  # 检查标签是否为 PyTorch 张量
        labels = torch.LongTensor(labels)  # 如果不是，将其转换为长整型张量
    preds = output.max(1)[1].type_as(labels)  # 获取预测结果中每个样本得分最高的类别索引，并转换为与标签相同的数据类型
    # correct = preds.eq(labels).double() # mps 不支持float64，原先使用double类型计算
    correct = preds.eq(labels).float()  # 判断预测结果与真实标签是否相等，得到布尔张量，并转换为浮点型（True为1.0，False为0.0）
    correct = correct.sum()  # 计算预测正确的样本数量
    return correct / len(labels)  # 返回准确率（正确数量 / 总样本数量）

def GCNAdjNorm(adj, order=-0.5):
    adj = sp.eye(adj.shape[0]) + adj
    # for i in range(len(adj.data)):
    #     if adj.data[i] > 0 and adj.data[i] != 1:
    #         adj.data[i] = 1
    adj.data[np.where((adj.data > 0) * (adj.data == 1))[0]] = 1
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, order).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return adj

def gcn_norm(adj_t, order=-0.5, add_self_loops=True):
    
    # if not adj_t.has_value():
    #     adj_t = adj_t.fill_value(1., dtype=None)
    if not isinstance(adj_t, SparseTensor):
        adj_t = SparseTensor.from_dense(adj_t)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.0)
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.float().pow_(order)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t

class RobustGCNConv(nn.Module):
    r"""
    Description
    -----------
    RobustGCN convolutional layer.
    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    act0 : func of torch.nn.functional, optional
        Activation function. Default: ``F.elu``.
    act1 : func of torch.nn.functional, optional
        Activation function. Default: ``F.relu``.
    initial : bool, optional
        Whether to initialize variance.
    dropout : bool, optional
        Whether to dropout during training. Default: ``False``.
    """

    def __init__(self, in_features, out_features, act0=F.elu, act1=F.relu, initial=False, dropout=0.5):
        super(RobustGCNConv, self).__init__()
        self.mean_conv = nn.Linear(in_features, out_features)
        self.var_conv = nn.Linear(in_features, out_features)
        self.act0 = act0
        self.act1 = act1
        self.initial = initial
        self.dropout = dropout

    def reset_parameters(self):
        self.mean_conv.reset_parameters()
        self.var_conv.reset_parameters()
    
    def forward(self, mean, var=None, adj0=None, adj1=None):
        r"""
        Parameters
        ----------
        mean : torch.Tensor
            Tensor of mean of input features.
        var : torch.Tensor, optional
            Tensor of variance of input features. Default: ``None``.
        adj0 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 0. Default: ``None``.
        adj1 : torch.SparseTensor, optional
            Sparse tensor of adjacency matrix 1. Default: ``None``.
        dropout : float, optional
            Rate of dropout. Default: ``0.0``.
        Returns
        -------
        """
        if self.initial:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var= mean
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var)
        else:
            mean = F.dropout(mean, p=self.dropout, training=self.training)
            var= F.dropout(var, p=self.dropout, training=self.training)
            mean = self.mean_conv(mean)
            var = self.var_conv(var)
            mean = self.act0(mean)
            var = self.act1(var)+1e-6 #avoid abnormal gradient
            attention = torch.exp(-var)
            mean = mean * attention
            var = var * attention * attention
            # print("adj0 mean",adj0,mean.shape)
            mean = adj0 @ mean
            var = adj1 @ var
            # print("mean1",mean.shape)
        return mean, var

class RobustGCN(nn.Module):
    r"""
    Description
    -----------
    Robust Graph Convolutional Networks (`RobustGCN <http://pengcui.thumedialab.com/papers/RGCN.pdf>`__)
    Parameters
    ----------
    in_features : int
        Dimension of input features.
    out_features : int
        Dimension of output features.
    hidden_features : int or list of int
        Dimension of hidden features. List if multi-layer.
    dropout : bool, optional
        Whether to dropout during training. Default: ``True``.
    """
    def __init__(self,
                 nfeat: int,
                 nhid: int,
                 nclass: int,
                 num_layers: int = 2,
                 dropout: float=0.5,
                 lr: float=0.01,
                 weight_decay: float=5e-4):
        # def __init__(self, in_features, out_features, hidden_features, dropout=True):
        super(RobustGCN, self).__init__()
        self.in_features = nfeat
        self.out_features = nclass

        self.act0 = F.elu
        self.act1 = F.relu

        self.layers = nn.ModuleList()
        self.layers.append(RobustGCNConv(nfeat, nhid, act0=self.act0, act1=self.act1,
                                         initial=True, dropout=dropout))
        for i in range(num_layers - 2):
            self.layers.append(RobustGCNConv(nhid, nhid,
                                             act0=self.act0, act1=self.act1, dropout=dropout))
        self.layers.append(RobustGCNConv(nhid, nclass, act0=self.act0, act1=self.act1))
        self.dropout = dropout
        self.use_ln = True
        self.gaussian = None
        
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay

    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index, return_embeddings=False):
        r"""
        Parameters
        ----------
        x : torch.Tensor
            Tensor of input features.
        edge_index : torch.Tensor
            Edge indices of the graph.
        return_embedding : bool, optional
            Whether to return embeddings along with output. Default: ``False``.
        Returns
        -------
        output : torch.Tensor
            Output of model (log softmax probabilities).
        embeddings : torch.Tensor, optional
            Node embeddings from the second-to-last layer (only if return_embedding=True).
        """
        num_nodes = x.shape[0]
        adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float)
        adj_matrix[edge_index[0], edge_index[1]] = 1.

        adj0, adj1 = gcn_norm(adj_matrix), gcn_norm(adj_matrix, order=-1.0)
        # adj0, adj1 = normalize_adj(adj), normalize_adj(adj, -1.0)
        mean = x
        var = x
        adj0 = adj0.to(x.device)
        adj1 = adj1.to(x.device)
        
        # Store embeddings from second-to-last layer
        embeddings = None
        
        for i, layer in enumerate(self.layers):
            # print(mean.shape,var.shape)
            mean, var = layer(mean, var=var, adj0=adj0, adj1=adj1)
            
            # Store embeddings from second-to-last layer
            if return_embeddings and i == len(self.layers) - 2:
                embeddings = mean.clone()
        
        # print(mean.shape,var.shape)
        # if self.gaussian == None:
        # self.gaussian = MultivariateNormal(torch.zeros(var.shape),
        #         torch.diag_embed(torch.ones(var.shape)))
        sample = torch.randn(var.shape).to(x.device)
        # sample = self.gaussian.sample().to(x.device)
        output = mean + sample * torch.pow(var, 0.5)
        
        # Apply log softmax to get final output
        output = output.log_softmax(dim=-1)
        
        if return_embeddings:
            # If no second-to-last layer exists (single layer), use the mean before final transformation
            if embeddings is None:
                embeddings = mean
            return output, embeddings
        
        return output

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()




class GCNConv_GNNGuard(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(GCNConv_GNNGuard, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.tensor(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        # """Here I removed the self-loop because the self-loop already added in the att_coef function"""
        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)

        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))
        # edge_index = to_undirected(edge_index, x.size(0))  # add non-direct edges
        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,self.out_channels)

class GNNGuard(nn.Module):

    def __init__(self,
                 nfeat: int,
                 nhid: int,
                 nclass: int,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 lr: float = 0.01,
                 drop: bool = False,
                 weight_decay: float = 5e-4,
                 with_relu: bool = True,
                 use_ln: bool = False,  # Fixed type annotation: should be bool, not float
                 with_bias: bool = True):  # Fixed type annotation: should be bool, not float
        super(GNNGuard, self).__init__()

        # Store model architecture parameters
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = int(nclass)
        self.num_layers = num_layers  # Store for potential future use
        self.dropout = dropout
        self.lr = lr
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.use_ln = use_ln
        
        # Weight decay logic: set to 0 if no ReLU, otherwise use provided value
        if not with_relu:
            self.weight_decay = 0.0
        else:
            self.weight_decay = weight_decay
        
        # Initialize layer normalization if requested
        if use_ln:
            self.lns = nn.ModuleList([
                nn.LayerNorm(nfeat),
                nn.LayerNorm(nhid)
            ])
        
        # Remove unused parameters that were cluttering the class
        # Removed: output, best_model, best_output, adj_norm, features, 
        #          gate, test_value, drop (unused), hidden_sizes
        
        # Initialize GCN layers with proper bias setting
        self.gc1 = GCNConv_GNNGuard(nfeat, nhid, bias=with_bias)
        self.gc2 = GCNConv_GNNGuard(nhid, self.nclass, bias=with_bias)

    def forward(self, x, adj, return_embeddings=False):
        """Forward pass of GNNGuard model.
        
        Args:
            x: Input node features
            adj: Edge indices of the graph
            return_embedding: Whether to return embeddings along with output
            
        Returns:
            output: Log softmax probabilities
            embeddings: Node embeddings from the first layer (only if return_embedding=True)
        """

        if self.use_ln:
            x = self.lns[0](x)
        edge_weight = self.att_coef(x, adj)
        x = self.gc1(x, adj, edge_weight=edge_weight)
        x = F.relu(x)
        
        # Store embeddings from first layer for return_embedding
        embeddings = x.clone() if return_embeddings else None
        
        if self.use_ln:
            x = self.lns[1](x)
        
        edge_weight = self.att_coef(x, adj)
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj, edge_weight=edge_weight)

        output = F.log_softmax(x, dim=1)
        
        if return_embeddings:
            return output, embeddings
        
        return output

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    # def att_coef(self, fea, edge_index):
    #     # Remove .detach() to allow gradient flow during training
    #     sim = torch.cosine_similarity(fea[edge_index[0]], fea[edge_index[1]])
    #
    #     # Use soft thresholding instead of hard thresholding for better gradient flow
    #     # Apply sigmoid-based soft thresholding: smooth transition around threshold
    #     threshold = 0.1
    #     steepness = 10.0  # Controls the steepness of the transition
    #
    #     # Soft thresholding: sigmoid((sim - threshold) * steepness) * sim
    #     # This creates a smooth transition instead of hard cutoff
    #     soft_mask = torch.sigmoid((sim - threshold) * steepness)
    #     sim = sim * soft_mask
    #
    #     return sim
    def att_coef(self, fea, edge_index):
        fea = fea.detach()
        sim = torch.cosine_similarity(fea[edge_index[0]], fea[edge_index[1]])
        sim[sim<0.1] = 0.0
        return sim


class ModelTrainer:
    """Trainer class for GNN models."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, 
                   data,
                   optimizer,
                   criterion,
                   train_mask):
        """Train for one epoch."""
        data = data.to(self.device)

        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data.x, data.edge_index)

        # Compute loss only on training nodes
        loss = criterion(out[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def evaluate(self, data, mask):
        """Evaluate model on given mask."""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(data, 'edge_index'):
                out = self.model(data.x, data.edge_index)
            else:
                out = self.model(data.x)
            
            pred = out[mask].argmax(dim=1)
            correct = (pred == data.y[mask]).float().sum()
            accuracy = correct / mask.sum()
        
        return accuracy.item()
    
    def get_predictions(self, data, return_probs=False):
        """Get model predictions."""
        self.model.eval()
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)

            if return_probs:
                probs = F.softmax(out, dim=1)
                return out, probs
            else:
                return out
    
    def get_embeddings(self, data):
        """Get node embeddings from the model."""
        self.model.eval()
        
        with torch.no_grad():
            if hasattr(data, 'edge_index'):
                _, embeddings = self.model(data.x, data.edge_index, return_embeddings=True)
            else:
                _, embeddings = self.model(data.x, return_embeddings=True)
        
        return embeddings




def create_model(model_name: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 **kwargs) -> nn.Module:
    """Factory function to create models.

    Args:
        model_name: Name of the model to create
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (number of classes)
        **kwargs: Additional model-specific arguments

    Returns:
        Initialized model
    """
    model_name = model_name.upper()

    if model_name == "GCN":
        return GCN(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "GRAPHSAGE":
        return GraphSAGE(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "GAT":
        return GAT(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "MLP":
        return MLP(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "ROBUSTGCN":
        return RobustGCN(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "GNNGUARD":
        return GNNGuard(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "GIN":
        return GIN(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "CHEBNET":
        return ChebNet(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "APPNPNET":
        return APPNPNet(input_dim, hidden_dim, output_dim, **kwargs)
    elif model_name == "GRAPHTRANSFORMER":
        return GraphTransformer(input_dim, hidden_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")