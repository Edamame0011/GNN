#pragma once

#include <torch/torch.h>

namespace FlashSchNet::kernels {
    std::tuple<torch::Tensor, torch::Tensor> fused_distance_gaussian_rbf_cutoff(
        torch::Tensor pos, 
        torch::Tensor edge_src, 
        torch::Tensor edge_dst, 
        torch::Tensor centers, 
        float gamma, 
        float cutoff
    );

    void fused_distance_gaussian_rbf_cutoff_grad_pos(
        torch::Tensor pos,
        torch::Tensor edge_src,
        torch::Tensor edge_dst,
        torch::Tensor centers,
        torch::Tensor distances,
        torch::Tensor grad_distances,
        torch::Tensor grad_rbf,
        torch::Tensor grad_pos,
        float gamma,
        float cutoff
    );

    torch::Tensor fused_csr_cfconv(
        torch::Tensor x, 
        torch::Tensor filter_out, 
        torch::Tensor edge_weight, 
        torch::Tensor edge_src, 
        torch::Tensor dst_ptr, 
        torch::Tensor csr_perm, 
        int num_nodes, 
        float cutoff
    );

    torch::Tensor fused_src_csr_grad_x(
        torch::Tensor grad_output, 
        torch::Tensor filter_out, 
        torch::Tensor edge_weight, 
        torch::Tensor edge_dst, 
        torch::Tensor src_ptr, 
        torch::Tensor src_perm, 
        int num_nodes, 
        float cutoff
    );

    torch::Tensor fused_grad_filter_out(
        torch::Tensor x, 
        torch::Tensor grad_output, 
        torch::Tensor edge_weight, 
        torch::Tensor edge_src, 
        torch::Tensor edge_dst, 
        float cutoff
    );
}