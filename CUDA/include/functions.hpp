#pragma once

#include <torch/torch.h>
#include <tuple>

namespace FlashSchNet::functions {
    class FusedDistanceGaussianRBFCutoffFunction : public torch::autograd::Function<FusedDistanceGaussianRBFCutoffFunction> {
        public:
            static std::vector<torch::Tensor> forward(
                torch::autograd::AutogradContext *ctx, 
                torch::Tensor pos, 
                torch::Tensor edge_src, 
                torch::Tensor edge_dst, 
                torch::Tensor centers, 
                float gamma, 
                float cutoff
            );

            static torch::autograd::variable_list backward(
                torch::autograd::AutogradContext* ctx, 
                torch::autograd::variable_list grad_outputs
            );
    };

    class FusedCSRCFConvFunction : public torch::autograd::Function<FusedCSRCFConvFunction> {
        public:
            static torch::Tensor forward(
                torch::autograd::AutogradContext *ctx, 
                torch::Tensor x, 
                torch::Tensor filter_out, 
                torch::Tensor edge_weight, 
                torch::Tensor edge_src, 
                torch::Tensor edge_dst, 
                torch::Tensor dst_ptr, 
                torch::Tensor csr_perm, 
                int num_nodes, 
                float cutoff, 
                torch::Tensor src_ptr, 
                torch::Tensor src_perm
            );

            static torch::autograd::variable_list backward(
                torch::autograd::AutogradContext* ctx, 
                torch::autograd::variable_list grad_outputs
            );
    };
}