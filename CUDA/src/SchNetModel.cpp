#include <FlashSchNet/layers/layers.hpp>
#include <FlashSchNet/layers/functions.hpp>
#include <FlashSchNet/utils/GraphBuilder.hpp>

using namespace FlashSchNet;

InteractionLayerImpl::InteractionLayerImpl(
    int hidden_dim, 
    int num_gaussians, 
    int num_filters, 
    float cutoff
) {
    this->mlp = register_module("mlp", torch::nn::Sequential(
        torch::nn::Linear(num_gaussians, num_filters), 
        ShiftedSoftplus(), 
        torch::nn::Linear(num_filters, num_filters)
    ));
    this->lin1 = register_module("lin1", torch::nn::Linear(
        torch::nn::LinearOptions(hidden_dim, num_filters).bias(false)
    ));
    this->lin2 = register_module("lin2", torch::nn::Linear(num_filters, hidden_dim));
    this->act = register_module("act", ShiftedSoftplus());
}

torch::Tensor InteractionLayerImpl::forward(
    torch::Tensor x, 
    torch::Tensor rbf_expansion, 
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
    torch::Tensor filter_out = mlp->forward(rbf_expansion);
    torch::Tensor x_v = lin1->forward(x);

    torch::Tensor conv_out = functions::FusedCSRCFConvFunction::apply(
        x_v, 
        filter_out, 
        edge_weight, 
        edge_src, 
        edge_dst, 
        dst_ptr, 
        csr_perm, 
        num_nodes, 
        cutoff, 
        src_ptr, 
        src_perm
    );

    torch::Tensor h = act->forward(lin2->forward(conv_out));

    return x + h;
}

SchNetModelImpl::SchNetModelImpl(
    int hidden_dim, 
    int num_gaussians, 
    float _cutoff, 
    int _num_filters, 
    int num_interactions, 
    int type_num = 100
) : cutoff(_cutoff), num_filters(_num_filters) {
    this->embedding = register_module(
        "embedding", 
        torch::nn::Embedding(type_num, hidden_dim)
    );
    this->interactions = register_module("interactions", torch::nn::ModuleList());
    for (size_t i = 0; i < num_interactions; i ++) {
        interactions->push_back(InteractionLayer(hidden_dim, num_gaussians, num_filters, cutoff));
    }
    this->output = register_module("output", torch::nn::Sequential(
        torch::nn::Linear(hidden_dim, hidden_dim  / 2), 
        ShiftedSoftplus(), 
        torch::nn::Linear(hidden_dim / 2, 1)
    ));
    torch::Tensor raw_centers = torch::linspace(0.0, cutoff, num_gaussians);
    this->centers = register_buffer("centers", raw_centers);
    float spacing = cutoff / std::max(1, (num_gaussians - 1));
    this->gamma = -0.5f / (spacing * spacing);
}

std::tuple<torch::Tensor, torch::Tensor> SchNetModelImpl::forward(
    torch::Tensor x, 
    torch::Tensor pos, 
    torch::Tensor edge_src, 
    torch::Tensor edge_dst, 
    utils::GraphBuilder* builder
) {
    int num_nodes = x.size(0);
    int num_edges = edge_src.size(0);

    if (!pos.requires_grad()) {
        pos.requires_grad_(true);
    }

    auto opt = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kInt64);
torch::Tensor dst_ptr = torch::empty({num_nodes + 1}, opt);
    torch::Tensor dst_perm = torch::empty({num_edges}, opt);
    torch::Tensor src_ptr = torch::empty({num_nodes + 1}, opt);
    torch::Tensor src_perm = torch::empty({num_edges}, opt);
    builder->build_csr_graph(edge_dst, dst_ptr, dst_perm, num_nodes);
    builder->build_csr_graph(edge_src, src_ptr, src_perm, num_nodes);

    auto h = embedding(x);

    auto [distances, rbf_expansion] = functions::FusedDistanceGaussianRBFCutoffFunction::apply(
        pos, 
        edge_src, 
        edge_dst, 
        centers, 
        gamma, 
        cutoff
    );

    for (auto& interaction : *interactions) {
        h = interaction->as<InteractionLayer>()->forward(
            h, 
            rbf_expansion, 
            distances, 
            edge_src, 
            edge_dst, 
            dst_ptr, 
            dst_perm, 
            num_nodes, 
            cutoff, 
            src_ptr, 
            src_perm
        );
    }

    auto energy = output->forward(h);
    auto total_energy = energy.sum();
    auto grads = torch::autograd::grad(
        {total_energy}, {pos}, {}, false, false
    );
    torch::Tensor forces = -grads[0];

    return std::make_tuple(total_energy, forces);
}