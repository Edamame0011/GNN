import torch
import torch.nn as nn
from model.schnet_model import SchNet_dict, GaussianRBF, ShiftedSoftplus, TypeEmbedding, CutoffFunction
from typing import Optional
from utils.MyGraphFunction import fused_distance_gaussian_rbf_cutoff, fused_csr_cfconv

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
            rbf_expansion: torch.Tensor, 
            edge_weight: torch.Tensor, 
            edge_src: torch.Tensor, 
            edge_dst: torch.Tensor, 
            dst_ptr: torch.Tensor, 
            num_nodes: int, 
            cutoff: float
        ):
        #フィルターの生成
        filter_out = self.mlp(rbf_expansion)
        x_v = self.lin1(x)

        conv_out = torch.ops.flash_schnet_ext.fused_csr_cfconv(
            x_v, 
            filter_out, 
            edge_weight, 
            edge_src, 
            edge_dst, 
            dst_ptr, 
            num_nodes, 
            cutoff
        )

        h = self.act(self.lin2(conv_out))

        return x + h

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
        self.register_buffer('centers', torch.linspace(0.0, cutoff, num_gaussians), persistent=False)
        spacing = cutoff / max(1, (num_gaussians - 1))
        self.gamma = -0.5 / (spacing * spacing)

    def forward(
            self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_weight: torch.Tensor, 
            dst_ptr: torch.Tensor
        ):
        edge_weight.requires_grad_(True)

        #埋め込み
        h = self.embedding(x) #(N, hidden_dim)

        i = edge_index[0]
        j = edge_index[1]

        distances = torch.norm(edge_weight, dim=0)

        rbf_expansion = fused_distance_gaussian_rbf_cutoff(
            distances, self.centers, self.gamma, self.cutoff
        )

        num_nodes = x.size(0)

        #Interactionレイヤー
        for interaction in self.interactions:
            h = interaction(h, rbf_expansion, distances, i, j, dst_ptr, num_nodes, self.cutoff)

        #各粒子のエネルギー
        energy = self.output(h) #(N, 1)
        total_energy = energy.sum()

        diff_E = torch.autograd.grad([total_energy], [edge_weight], create_graph=False)[0] #(3, num_edges)
        assert diff_E is not None

        #forces: 力を受ける側の粒子が受ける力 (3, N)
        forces = x.new_zeros(3, x.size(0), dtype=edge_weight.dtype)
        forces.index_add_(1, i, diff_E)
        forces.index_add_(1, j, -diff_E)
        
        return total_energy, forces