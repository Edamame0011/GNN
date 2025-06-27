import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
import numpy as np
from typing import Optional

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

#Interactionレイヤー
class InterectionBlock(nn.Module):
    def __init__(self, hidden_dim, num_gaussians, num_filters, cutoff):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_gaussians, num_filters), 
            ShiftedSoftplus(), 
            nn.Linear(num_filters, num_filters)
        )
        self.cutoff = cutoff
        self.cutoff_function = CutoffFunction()
        self.lin1 = nn.Linear(hidden_dim, num_filters, bias = False)
        self.lin2 = nn.Linear(num_filters, hidden_dim)
        self.act = ShiftedSoftplus()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, edge_attr: torch.Tensor):
        #原子間距離
        distances = torch.norm(edge_weight, dim = 1) #(num_edges, )
        #原子間距離のカットオフ
        C = self.cutoff_function(distances, self.cutoff) #(num_edges, )

        #フィルターの生成
        W = self.mlp(edge_attr) * C.unsqueeze(-1) #(num_edges, num_filters)
        #C.unsqueeze(-1): (num_edges, ) -> (num_edges, 1)
        #mlp: (num_edges, num_gaussians) -> (num_edges, num_filters)

        #メッセージ生成
        i = edge_index[0]
        j = edge_index[1]       
        #i: 送信先のノードのインデックス (num_edges, )
        #j: 送信元のノードのインデックス (num_edges, )

        messages = W * self.lin1(x[j]) #(num_edges, num_filters)
        #x: (num_nodes, hidden_dim)
        #x[j]: (num_edges, hidden_dim)
        #lin1: (hidden_dim, ) -> (num_filters, )
        #message[k]: ノードi[k]に届くメッセージを表す。

        #メッセージ集約
        #agg_message[k]: ノードkに届くメッセージを表す。
        agg_messages = torch.zeros_like(self.lin1(x))
        index = i.unsqueeze(1) if i.ndim == 1 else i
        agg_messages = torch.scatter_add(agg_messages, dim = 0, index = index.expand_as(messages), src = messages)

        #特徴量更新
        h = self.act(self.lin2(agg_messages))

        return x + h #残差接続

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

#モデル
class SchNetModel(nn.Module):
    def __init__(self, hidden_dim, num_gaussians, cutoff, num_filters, num_interactions, type_num = 100):
        super().__init__()

        self.setups = SchNet_dict(hidden_dim, num_gaussians, num_filters, num_interactions, cutoff, type_num)

        self.embedding = TypeEmbedding(type_num, hidden_dim)
        self.rbf = GaussianRBF(num_gaussians, cutoff)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InterectionBlock(hidden_dim, num_gaussians, num_filters, cutoff)
            self.interactions.append(block)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            ShiftedSoftplus(), 
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: Optional[torch.Tensor] = None):
        edge_weight.requires_grad_()

        #埋め込み
        h = self.embedding(x) #(num_nodes, hidden_dim)

        #RBF展開
        distances = torch.norm(edge_weight, dim = 1)
        rbf_expansion = self.rbf(distances)

        #Interactionレイヤー
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_weight, rbf_expansion)
        
        #各粒子のエネルギー
        energy = self.output(h) #(num_nodes, 1)

        diff_E = torch.autograd.grad([energy.sum()], [edge_weight], create_graph = True)[0] #(num_edges, 3)
        assert diff_E is not None   #TorchScriptに明示的にNoneではないことを伝える

        #i: 送信先のノードのインデックス (num_edges, )
        #j: 送信元のノードのインデックス (num_edges, )
        i = edge_index[0]
        j = edge_index[1]      

        #力を加算。この際、作用反作用の法則から、粒子iにはdiffEが、粒子jには-diffEがかかる。
        #force_i: 力を受ける側の粒子が受ける力 (num_nodes, 3)
        #force_j: 力を与える側の粒子が受ける力 (num_nodes, 3)
        force_i = torch.zeros((len(x), 3), device = edge_weight.device)
        force_j = torch.zeros((len(x), 3), device = edge_weight.device)
        index_i = i.unsqueeze(1) if i.ndim == 1 else i
        index_j = j.unsqueeze(1) if j.ndim == 1 else j
        force_i = torch.scatter_add(force_i, dim = 0, index = index_i.expand_as(diff_E), src = diff_E)
        force_j = torch.scatter_add(force_j, dim = 0, index = index_j.expand_as(diff_E), src = -diff_E)

        #それぞれの粒子が受ける力
        forces = force_i + force_j #(num_nodes, 3)

        #バッチごとに集約
        #ここで、batchは、それぞれのノードが所属するサンプル番号を表す1次元テンソル。ノードkは、batch[k]に属する。
        if batch is not None:
            batch_max = batch.max()
            #total_energy[k]: k番目のサンプルのエネルギー
            total_energy = torch.zeros(batch_max + 1, device = energy.device)
            total_energy = total_energy.index_add_(0, batch, energy.squeeze())
        
        else:
            total_energy = energy.sum() #全ノードの合計エネルギー
        
        return total_energy, forces