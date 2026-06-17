#include "functions.hpp"
#include "kernels.hpp"

namespace FlashSchNet::functions {
    std::vector<torch::Tensor> FusedDistanceGaussianRBFCutoffFunction::forward(
        torch::autograd::AutogradContext *ctx, 
        torch::Tensor pos, 
        torch::Tensor edge_src, 
        torch::Tensor edge_dst, 
        torch::Tensor centers, 
        float gamma, 
        float cutoff
    ) {
        auto [distances, rbf_expansion] = kernels::fused_distance_gaussian_rbf_cutoff(pos, edge_src, edge_dst, centers, gamma, cutoff);
        ctx->save_for_backward({pos, edge_src, edge_dst, centers, distances});
        ctx->saved_data["gamma"] = gamma;
        ctx->saved_data["cutoff"] = cutoff;
        return {distances, rbf_expansion};
    }

    torch::autograd::variable_list FusedDistanceGaussianRBFCutoffFunction::backward(
        torch::autograd::AutogradContext* ctx, 
        torch::autograd::variable_list grad_outputs
    ) {
        auto grad_distances = grad_outputs[0].contiguous();
        auto grad_rbf = grad_outputs[1].contiguous();

        auto saved = ctx->get_saved_variables();
        auto pos = saved[0];
        auto edge_src = saved[1];
        auto edge_dst = saved[2];
        auto centers = saved[3];
        auto distances = saved[4];

        float gamma = ctx->saved_data["gamma"].toDouble();
        float cutoff = ctx->saved_data["cutoff"].toDouble();

        auto grad_pos = torch::zeros_like(pos);

        if(ctx->needs_input_grad(0)) {
            kernels::fused_distance_gaussian_rbf_cutoff_grad_pos(
                pos, edge_src, edge_dst, centers, distances, 
                grad_distances, grad_rbf, grad_pos, 
                gamma, cutoff
            );
        }

        return {
            grad_pos, 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor()
        };
    }

    torch::Tensor FusedCSRCFConvFunction::forward(
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
    ) {
        ctx->save_for_backward({x, filter_out, edge_weight, edge_src, edge_dst, dst_ptr, csr_perm, src_ptr, src_perm});
        ctx->saved_data["num_nodes"] = num_nodes;
        ctx->saved_data["cutoff"] = cutoff;

        auto out = kernels::fused_csr_cfconv(x, filter_out, edge_weight, edge_src, dst_ptr, csr_perm, num_nodes, cutoff);

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
        auto csr_perm = saved[6];
        auto src_ptr = saved[7];
        auto src_perm = saved[8];

        int num_nodes = ctx->saved_data["num_nodes"].toInt();
        float cutoff = (float)ctx->saved_data["cutoff"].toDouble();

        torch::Tensor grad_x, grad_filter_out;

        if (ctx->needs_input_grad(0)) {
            grad_x = kernels::fused_src_csr_grad_x(
                grad_output, 
                filter_out, 
                edge_weight, 
                edge_dst, 
                src_ptr, 
                src_perm, 
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

        return {
            grad_x, 
            grad_filter_out, 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor(), 
            torch::Tensor()
        };
    }
}