import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi as PI
import numpy as np
from typing import Optional
from utils.MyGraphFunction import message_aggregate, calc_force
from model.schnet_model_CSR import SchNet_dict, GaussianRBF, InteractionBlock, ShiftedSoftplus, TypeEmbedding, CutoffFunction

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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, offsets: torch.Tensor):
        """
        x: (num_nodes, )
        edge_weight: (3, num_edges)
        edge_index: (2, num_edges)
        """
        
        i, j = edge_index[0], edge_index[1]
        # i: target node index
        # j: source node index

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

        total_energy = energy.sum()

        # 力の計算
        diff_E = torch.autograd.grad(
            outputs=[total_energy], 
            inputs=[edge_weight]
        )[0]
        assert diff_E is not None

        #forces: 力を受ける側の粒子が受ける力 (3, N)
        forces = x.new_zeros(3, x.size(0), dtype=edge_weight.dtype)
        forces.index_add_(1, i, diff_E)
        forces.index_add_(1, j, -diff_E)
        
        return total_energy, forces