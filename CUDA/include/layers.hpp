#pragma once

#include <torch/torch.h>

namespace FlashSchNet{
    namespace utils {
        class GraphBuilder;
    }

    struct ShiftedSoftplusImpl : torch::nn::Module {
        torch::Tensor log2;

        ShiftedSoftplusImpl() {
            log2 = register_buffer("log2", torch::log(torch::tensor(2.0)));
        }
        torch::Tensor forward(const torch::Tensor& x) {
            return torch::nn::functional::softplus(x) - log2;
        }
    };
    TORCH_MODULE(ShiftedSoftplus);

    class InteractionLayerImpl : public torch::nn::Module {
        public:
            InteractionLayerImpl(
                int hidden_dim, 
                int num_gaussians, 
                int num_filters, 
                float cutoff
            );
            torch::Tensor forward(
                torch::Tensor x, 
                torch::Tensor edge_attr, 
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
        private:
            torch::nn::Linear lin1{nullptr}, lin2{nullptr};
            torch::nn::Sequential mlp{nullptr};
            ShiftedSoftplus act{nullptr};
    };
    TORCH_MODULE(InteractionLayer);

    class SchNetModelImpl : public torch::nn::Module {
        public:
            SchNetModelImpl(
                int hidden_dim, 
                int num_gaussians, 
                float cutoff, 
                int num_filters, 
                int num_interactions, 
                int type_num = 100
            );
            std::tuple<torch::Tensor, torch::Tensor> forward(
                torch::Tensor x, 
                torch::Tensor pos, 
                torch::Tensor edge_src, 
                torch::Tensor edge_dst, 
                utils::GraphBuilder* builder
            );

        private:
            float cutoff; 
            int num_filters;
            torch::Tensor centers;
            float gamma;
            torch::nn::Embedding embedding{nullptr};
            torch::nn::ModuleList interactions{nullptr};
            torch::nn::Sequential output{nullptr};
    };
    TORCH_MODULE(SchNetModel);
}