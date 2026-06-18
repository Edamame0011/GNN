#pragma once

#include <torch/torch.h>

namespace FlashSchNet::kernels {
    torch::Tensor fused_distance_gaussian_rbf_cutoff(
        const torch::Tensor& edge_weight, 
        const torch::Tensor& centers, 
        float gamma, 
        float cutoff
    );

    torch::Tensor fused_distance_gaussian_rbf_cutoff_grad_pos(
        const torch::Tensor& edge_weight,
        const torch::Tensor& centers,
        const torch::Tensor& grad_rbf, 
        float gamma,
        float cutoff
    );

    torch::Tensor fused_csr_cfconv(
        const torch::Tensor& x, 
        const torch::Tensor& filter_out, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& dst_ptr, 
        float cutoff
    );

    torch::Tensor fused_csr_grad_x(
        const torch::Tensor& grad_output, 
        const torch::Tensor& filter_out, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& edge_dst, 
        float cutoff
    );

    torch::Tensor fused_grad_filter_out(
        const torch::Tensor& x, 
        const torch::Tensor& grad_output, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& edge_dst, 
        float cutoff
    );

    torch::Tensor fused_grad_edge_weight(
        const torch::Tensor& grad_output, 
        const torch::Tensor& x, 
        const torch::Tensor& filter_out, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& dst_ptr, 
        float cutoff
    );
}