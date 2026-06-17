#pragma once

#include <torch/torch.h>
#include <tuple>

namespace FlashSchNet::functions {
    class FusedDistanceGaussianRBFCutoffFunction : public torch::autograd::Function<FusedDistanceGaussianRBFCutoffFunction> {
        public:
            static torch::Tensor forward(
                torch::autograd::AutogradContext *ctx, 
                const torch::Tensor& edge_weight, 
                const torch::Tensor& centers, 
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
                const torch::Tensor& x, 
                const torch::Tensor& filter_out, 
                const torch::Tensor& edge_weight, 
                const torch::Tensor& edge_src, 
                const torch::Tensor& edge_dst, 
                const torch::Tensor& dst_ptr, 
                int num_nodes, 
                float cutoff
            );

            static torch::autograd::variable_list backward(
                torch::autograd::AutogradContext* ctx, 
                torch::autograd::variable_list grad_outputs
            );
    };
}