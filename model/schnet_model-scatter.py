import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
import numpy as np
from torch_scatter import scatter
from typing import Tuple, Optional

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

    def forward(self, x: torch.Tensor, edge_index: Tuple[torch.Tensor, torch.Tensor], edge_weight: torch.Tensor, edge_attr: torch.Tensor):
        #原子間距離
        distances = torch.norm(edge_weight, dim = 1) #(num_edges, )
        #原子間距離のカットオフ
        C = self.cutoff_function(distances, self.cutoff) #(num_edges, )

        #フィルターの生成
        W = self.mlp(edge_attr) * C.unsqueeze(-1) #(num_edges, num_filters)
        #C.unsqueeze(-1): (num_edges, ) -> (num_edges, 1)
        #mlp: (num_edges, num_gaussians) -> (num_edges, num_filters)

        #メッセージ生成
        i, j = edge_index       
        #i: 送信先のノードのインデックス (num_edges, )
        #j: 送信元のノードのインデックス (num_edges, )

        messages = W * self.lin1(x[j]) #(num_edges, num_filters)
        #x: (num_nodes, hidden_dim)
        #x[j]: (num_edges, hidden_dim)
        #lin1: (hidden_dim, ) -> (num_filters, )
        #message[k]: ノードi[k]に届くメッセージを表す。

        #メッセージ集約
        #agg_message[k]: ノードkに届くメッセージを表す。
        agg_messages = scatter(messages, i, dim = 0, reduce = 'add')
        #scatter(src, index, dim, dim_size, reduce)
        #scr: 集約元のテンソル（足されるもの）
        #index: srcと同じ形状の、srcの要素をグループに分けるようなインデックス
        #dim: srcとindexの沿う軸番号（形状が同じ要素の番号）
        #例えば、src = ([2, 4, 6], [3, 1, 2], [6, 8, 4], [2, 3, 9], [9, 8, 4], [1, 3 ,2]), index = (0, 1, 0, 2, 2, 1)の時、
        #scatter(src, index, dim=0) = ([8, 12, 10], [4, 4, 4], [11, 11, 13])
        #scatter()[0]が0グループ目の足し算の結果、scatter()[1]が、、、、

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

    def forward(self, x: torch.Tensor, edge_index: Tuple[torch.Tensor, torch.Tensor], edge_weight: torch.Tensor, batch: Optional[torch.Tensor] = None):
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
        i, j = edge_index      

        #力を加算。この際、作用反作用の法則から、粒子iにはdiffEが、粒子jには-diffEがかかる。
        #force_i: 力を受ける側の粒子が受ける力 (num_nodes, 3)
        #force_j: 力を与える側の粒子が受ける力 (num_nodes, 3)
        force_i = scatter(diff_E, i, dim = 0, dim_size = len(x), reduce = 'add')
        force_j = scatter(-diff_E, j, dim = 0, dim_size = len(x), reduce = 'add')

        #それぞれの粒子が受ける力
        forces = force_i + force_j #(num_nodes, 3)

        #バッチごとに集約
        #ここで、batchは、それぞれのノードが所属するサンプル番号を表す1次元テンソル。ノードkは、batch[k]に属する。
        if batch is not None:
            #total_energy[k]: k番目のサンプルのエネルギー
            total_energy = scatter(energy.squeeze(), batch, dim = 0, reduce = 'add')
        
        else:
            total_energy = energy.sum() #全ノードの合計エネルギー
        
        return total_energy, forces