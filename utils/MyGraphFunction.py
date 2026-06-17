import torch
import flash_schnet_ext

def fused_distance_gaussian_rbf_cutoff(
        edge_weight: torch.Tensor, 
        centers: torch.Tensor, 
        gamma: float, 
        cutoff: float
    ) -> torch.Tensor:
    # torch.ops 経由で呼び出す
    return torch.ops.flash_schnet_ext.fused_distance_gaussian_rbf_cutoff(
        edge_weight, centers, gamma, cutoff
    )

def fused_csr_cfconv(
        x: torch.Tensor, 
        filter_out: torch.Tensor, 
        edge_weight: torch.Tensor, 
        edge_src: torch.Tensor, 
        edge_dst: torch.Tensor, 
        dst_ptr: torch.Tensor, 
        num_nodes: int, 
        cutoff: float
    ) -> torch.Tensor:
    # torch.ops 経由で呼び出す
    return torch.ops.flash_schnet_ext.fused_csr_cfconv(
        x, filter_out, edge_weight, edge_src, edge_dst, dst_ptr, num_nodes, cutoff
    )