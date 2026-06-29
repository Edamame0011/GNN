import torch
import flash_schnet_ext

@torch.library.register_fake("flash_schnet_ext::fused_distance_gaussian_rbf_cutoff")
def _fused_distance_gaussian_rbf_cutoff_fake(
    edge_weight: torch.Tensor, 
    centers: torch.Tensor, 
    gamma: float, 
    cutoff: float
) -> torch.Tensor:
    num_edges = edge_weight.shape[0]
    num_rbf = centers.shape[0]
    
    # 計算はせず、出力と同じ形状・型・デバイスを持つ空のテンソルを返す
    return torch.empty((num_edges, num_rbf), dtype=edge_weight.dtype, device=edge_weight.device)


@torch.library.register_fake("flash_schnet_ext::fused_csr_cfconv")
def _fused_csr_cfconv_fake(
    x: torch.Tensor, 
    filter_out: torch.Tensor, 
    edge_weight: torch.Tensor, 
    edge_src: torch.Tensor, 
    edge_dst: torch.Tensor, 
    dst_ptr: torch.Tensor, 
    cutoff: float
) -> torch.Tensor:
    # x と全く同じ形状・型・デバイスの空テンソルを返す
    return torch.empty_like(x)

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
        cutoff: float
    ) -> torch.Tensor:
    # torch.ops 経由で呼び出す
    return torch.ops.flash_schnet_ext.fused_csr_cfconv(
        x, filter_out, edge_weight, edge_src, edge_dst, dst_ptr, cutoff
    )