#include "functions.hpp"
#include "kernels.hpp"

namespace FlashSchNet::functions {
    torch::Tensor FusedDistanceGaussianRBFCutoffFunction::forward(
        torch::autograd::AutogradContext *ctx, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& centers, 
        float gamma, 
        float cutoff
    ) {
        auto rbf_expansion = kernels::fused_distance_gaussian_rbf_cutoff(
            edge_weight, 
            centers, 
            gamma, 
            cutoff
        );
        ctx->save_for_backward({edge_weight, centers});
        ctx->saved_data["gamma"] = gamma;
        ctx->saved_data["cutoff"] = cutoff;
        return rbf_expansion;
    }

    torch::autograd::variable_list FusedDistanceGaussianRBFCutoffFunction::backward(
        torch::autograd::AutogradContext* ctx, 
        torch::autograd::variable_list grad_outputs
    ) {
        auto grad_rbf = grad_outputs[0].contiguous();

        auto saved = ctx->get_saved_variables();
        auto edge_weight = saved[0];
        auto centers = saved[1];

        float gamma = ctx->saved_data["gamma"].toDouble();
        float cutoff = ctx->saved_data["cutoff"].toDouble();
    
        torch::Tensor grad_edge_weight;

        if (ctx->needs_input_grad(0)) {
            grad_edge_weight = kernels::fused_distance_gaussian_rbf_cutoff_grad_pos(
                edge_weight, 
                centers, 
                grad_rbf, 
                gamma, 
                cutoff
            );
        }

        return {
            grad_edge_weight, 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor()
        };
    }

    torch::Tensor FusedCSRCFConvFunction::forward(
        torch::autograd::AutogradContext *ctx, 
        const torch::Tensor& x, 
        const torch::Tensor& filter_out, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& edge_dst, 
        const torch::Tensor& dst_ptr, 
        int num_nodes, 
        float cutoff
    ) {
        ctx->save_for_backward({x, filter_out, edge_weight, edge_src, edge_dst, dst_ptr});
        ctx->saved_data["num_nodes"] = num_nodes;
        ctx->saved_data["cutoff"] = cutoff;

        auto out = kernels::fused_csr_cfconv(x, filter_out, edge_weight, edge_src, dst_ptr, num_nodes, cutoff);

        return out;
    }

    torch::autograd::variable_list FusedCSRCFConvFunction::backward(
        torch::autograd::AutogradContext* ctx, 
        torch::autograd::variable_list grad_outputs
    ) {
        auto grad_output = grad_outputs[0].contiguous();

        auto saved = ctx->get_saved_variables();
        auto x = saved[0];
        auto filter_out = saved[1];
        auto edge_weight = saved[2];
        auto edge_src = saved[3];
        auto edge_dst = saved[4];
        auto dst_ptr = saved[5];

        int num_nodes = ctx->saved_data["num_nodes"].toInt();
        float cutoff = (float)ctx->saved_data["cutoff"].toDouble();

        torch::Tensor grad_x, grad_filter_out, grad_edge_weight;

        if (ctx->needs_input_grad(0)) {
            grad_x = kernels::fused_csr_grad_x(
                grad_output, 
                filter_out, 
                edge_weight, 
                edge_src, 
                edge_dst, 
                num_nodes, 
                cutoff
            );
        }

        if (ctx->needs_input_grad(1)) {
            grad_filter_out = kernels::fused_grad_filter_out(
                x, 
                grad_output, 
                edge_weight, 
                edge_src, 
                edge_dst, 
                cutoff
            );
        }

        if (ctx->needs_input_grad(2)) {
            grad_edge_weight = kernels::fused_grad_edge_weight(
                grad_output, 
                x, 
                filter_out, 
                edge_weight, 
                edge_src, 
                dst_ptr, 
                num_nodes, 
                cutoff
            );
        }

        return {
            grad_x, 
            grad_filter_out, 
            grad_edge_weight, 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor()
        };
    }
}