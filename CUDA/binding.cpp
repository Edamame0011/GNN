#include <torch/extension.h>
#include <torch/script.h>

#include "functions.hpp"

using namespace FlashSchNet::functions;

#define CHECK_INT32(x) TORCH_CHECK(x.scalar_type() == torch::kInt32, #x " must be an int32 tensor")

torch::Tensor fused_distance_gaussian_rbf_cutoff(
    const torch::Tensor& edge_weight, 
    const torch::Tensor& centers, 
    double gamma, 
    double cutoff
) {
    return FusedDistanceGaussianRBFCutoffFunction::apply(
        edge_weight, centers, (float)gamma, (float)cutoff
    );
}

torch::Tensor fused_csr_cfconv(
    const torch::Tensor& x, 
    const torch::Tensor& filter_out, 
    const torch::Tensor& edge_weight, 
    const torch::Tensor& edge_src, 
    const torch::Tensor& edge_dst, 
    const torch::Tensor& dst_ptr, 
    int64_t num_nodes, 
    double cutoff
) {
    CHECK_INT32(edge_src);
    CHECK_INT32(edge_dst);
    CHECK_INT32(dst_ptr);

    return FusedCSRCFConvFunction::apply(
        x, filter_out, edge_weight, edge_src, edge_dst, dst_ptr, 
        (int)num_nodes, (float)cutoff
    );
}

TORCH_LIBRARY(flash_schnet_ext, m) {
    m.def("fused_distance_gaussian_rbf_cutoff(Tensor edge_weight, Tensor centers, float gamma, float cutoff) -> Tensor");
    m.def("fused_csr_cfconv(Tensor x, Tensor filter_out, Tensor edge_weight, Tensor edge_src, Tensor edge_dst, Tensor dst_ptr, int num_nodes, float cutoff) -> Tensor");
}

TORCH_LIBRARY_IMPL(flash_schnet_ext, Autograd, m) {
    m.impl("fused_distance_gaussian_rbf_cutoff", &fused_distance_gaussian_rbf_cutoff);
    m.impl("fused_csr_cfconv", &fused_csr_cfconv);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom Graph Operations for SchNet";
}