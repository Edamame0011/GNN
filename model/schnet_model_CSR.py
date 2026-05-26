import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
import numpy as np
from typing import Optional
from utils.MyGraphFunction import message_aggregate, calc_force

#ガウス基底関数
def gaussian_rbf(inputs, offsets, widths):
    coeff = -0.5 / widths ** 2
    diff = inputs[..., None] - offsets
    return torch.exp(coeff * diff ** 2)

class GaussianRBF(nn.Module):
    def __init__(self, n_rbf: int, cutoff: float, start = 0.0):
        super().__init__()
        self.register_buffer("offsets", torch.linspace(start, cutoff, n_rbf))
        self.register_buffer("widths", torch.full((n_rbf,), cutoff / n_rbf))

    def forward(self, distances: torch.Tensor):
        return gaussian_rbf(distances, self.offsets, self.widths)

#Shifted Softplus関数
class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("log2", torch.log(torch.tensor(2.0)))
    
    def forward(self, x: torch.Tensor):
        return F.softplus(x) - self.log2

#カットオフ関数
#distanceがcutoffに近づくにつれ滑らかに0に
class CutoffFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.PI = PI

    def forward(self, distances: torch.Tensor, cutoff: float):
        mask = distances <= cutoff
        C = 0.5 * (torch.cos(distances * self.PI / cutoff) + 1.0) * mask
        return C

#原子種類の埋め込みレイヤー
class TypeEmbedding(nn.Module):
    def __init__(self, type_num, type_dim):
        super().__init__()
        #(type_num, ) -> (type_dim, )
        self.embedding = nn.Embedding(type_num, type_dim)
    
    def forward(self, x: torch.Tensor):
        return self.embedding(x)

#設定を管理する辞書型配列
class SchNet_dict():
    def __init__(self, hidden_dim, num_gaussians, num_filters, num_interactions, cutoff, type_num = 100):
        self.hidden_dim = hidden_dim
        self.num_gaussians = num_gaussians
        self.num_filters = num_filters
        self.num_interactions = num_interactions
        self.cutoff = cutoff
        self.type_num = type_num
    
    def to_dict(self):
        return {
            "hidden_dim" : self.hidden_dim,
            "num_gaussians" : self.num_gaussians, 
            "num_filters" : self.num_filters,
            "num_interactions" : self.num_interactions,
            "cutoff" : self.cutoff, 
            "type_num" : self.type_num
        }
    
    @classmethod
    def from_dict(cls, dic):
        return cls(**dic)

#Interactionレイヤー
class InteractionBlock(nn.Module):
    def __init__(self, hidden_dim, num_gaussians, num_filters):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters), 
            ShiftedSoftplus(), 
            nn.Linear(num_filters, num_filters)
        )
        self.num_filters = num_filters
        self.lin1 = nn.Linear(hidden_dim, num_filters, bias = False)
        self.lin2 = nn.Linear(num_filters, hidden_dim)
        self.act = ShiftedSoftplus()

    def forward(
            self, 
            x: torch.Tensor, 
            offsets: torch.Tensor, 
            i: torch.Tensor, 
            j: torch.Tensor, 
            edge_attr: torch.Tensor, 
            C: torch.Tensor
        ):
        #フィルターの生成
        W = self.mlp(edge_attr) * C.unsqueeze(-1)

        #メッセージ生成
        messages = W * self.lin1(x[j])

        #メッセージ集約
        agg_messages = message_aggregate(
            messages, 
            offsets, 
            i, 
            x.size(0)
        )

        #特徴量更新
        h = self.act(self.lin2(agg_messages))

        return x + h #残差接続

#モデル
class SchNetModel(nn.Module):
    def __init__(self, hidden_dim, num_gaussians, cutoff, num_filters, num_interactions, type_num = 100):
        super().__init__()

        self.setups = SchNet_dict(hidden_dim, num_gaussians, num_filters, num_interactions, cutoff, type_num)

        self.embedding = TypeEmbedding(type_num, hidden_dim)
        self.rbf = GaussianRBF(num_gaussians, cutoff)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_dim, num_gaussians, num_filters)
            self.interactions.append(block)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            ShiftedSoftplus(), 
            nn.Linear(hidden_dim // 2, 1)
        )

        self.cutoff = cutoff
        self.cutoff_function = CutoffFunction()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: Optional[torch.Tensor] = None):
        """
        x: (num_nodes, )
        edge_weight: (3, num_edges)
        edge_index: (2, num_edges)
        """
        if not edge_weight.requires_grad:
            edge_weight.requires_grad_(True)
            
        num_nodes = x.size(0)
        
        i, j = edge_index[0], edge_index[1]
        # i: target node index
        # j: source node index

        degrees_dst = torch.bincount(i, minlength=num_nodes)
        offsets = torch.zeros(num_nodes + 1, dtype=torch.int64, device=x.device)
        offsets[1:] = torch.cumsum(degrees_dst, dim=0)

        distances = torch.norm(edge_weight, dim=0) # (num_edges, )

        #埋め込み
        h = self.embedding(x) #(N, hidden_dim)

        #RBF展開
        rbf_expansion = self.rbf(distances) # (num_edges, num_gaussians)

        #原子間距離のカットオフ
        C = self.cutoff_function(distances, self.cutoff) #(num_edges, )

        #Interactionレイヤー
        for interaction in self.interactions:
            h = interaction(h, offsets, i, j, rbf_expansion, C)
        
        #各粒子のエネルギー
        energy = self.output(h) #(N, 1)

        # 力の計算
        diff_E = torch.autograd.grad(
            outputs=energy.sum(), 
            inputs=edge_weight, 
            create_graph=True, 
            retain_graph=True
        )[0]

        #forces: 力を受ける側の粒子が受ける力 (3, N)
        forces = x.new_zeros(3, x.size(0), dtype=edge_weight.dtype)
        forces.index_add_(1, i, diff_E)
        forces.index_add_(1, j, -diff_E)

        #バッチごとに集約
        if batch is not None:
            batch_max = batch.max()
            total_energy = torch.zeros(batch_max + 1, device=energy.device)
            total_energy = total_energy.index_add_(0, batch, energy.squeeze(-1))
        
        else:
            total_energy = energy.sum()
        
        return total_energy, forces