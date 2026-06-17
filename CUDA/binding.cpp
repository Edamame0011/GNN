#include <torch/extension.h>
#include <torch/script.h>

#include "functions.hpp"

using namespace FlashSchNet::functions;

std::vector<torch::Tensor> fused_distance_gaussian_rbf_cutoff(
    torch::Tensor pos, 
    torch::Tensor edge_src, 
    torch::Tensor edge_dst, 
    torch::Tensor centers, 
    double gamma, 
    double cutoff) 
{
    return FusedDistanceGaussianRBFCutoffFunction::apply(
        pos, edge_src, edge_dst, centers, gamma, cutoff);
}

torch::Tensor fused_csr_cfconv(
    torch::Tensor x, 
    torch::Tensor filter_out, 
    torch::Tensor edge_weight, 
    torch::Tensor edge_src, 
    torch::Tensor edge_dst, 
    torch::Tensor dst_ptr, 
    torch::Tensor csr_perm, 
    int64_t num_nodes, 
    double cutoff, 
    torch::Tensor src_ptr, 
    torch::Tensor src_perm) 
{
    return FusedCSRCFConvFunction::apply(
        x, filter_out, edge_weight, edge_src, edge_dst, dst_ptr, 
        csr_perm, num_nodes, cutoff, src_ptr, src_perm);
}

TORCH_LIBRARY(flash_schnet_ext, m) {
    m.def("fused_distance_gaussian_rbf_cutoff(Tensor pos, Tensor edge_src, Tensor edge_dst, Tensor centers, float gamma, float cutoff) -> Tensor[]");
    m.def("fused_csr_cfconv(Tensor x, Tensor filter_out, Tensor edge_weight, Tensor edge_src, Tensor edge_dst, Tensor dst_ptr, Tensor csr_perm, int num_nodes, float cutoff, Tensor src_ptr, Tensor src_perm) -> Tensor");
}

TORCH_LIBRARY_IMPL(flash_schnet_ext, Autograd, m) {
    m.impl("fused_distance_gaussian_rbf_cutoff", &fused_distance_gaussian_rbf_cutoff);
    m.impl("fused_csr_cfconv", &fused_csr_cfconv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom Graph Operations for SchNet";
}