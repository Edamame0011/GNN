#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {
    __global__ void calc_force_forward_kernel(
        float* __restrict__ force, 
        const float* __restrict__ diff_E, 
        const int64_t* __restrict__ dst_node_ptr, 
        const int64_t num_nodes, 
        const int64_t num_edges
    ) {
        const int64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        const int64_t warp_id = tid / 32;
        const int lane_id = threadIdx.x % 32;
        if (warp_id >= num_nodes) return;

        const int64_t start_edge_idx = dst_node_ptr[warp_id];
        const int64_t end_edge_idx = dst_node_ptr[warp_id + 1];

        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        for (int64_t e = start_edge_idx + lane_id; e < end_edge_idx; e += 32) {
            fx += diff_E[e];
            fy += diff_E[num_edges + e];
            fz += diff_E[2 * num_edges + e];
        }

        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            fx += __shfl_down_sync(mask, fx, offset);
            fy += __shfl_down_sync(mask, fy, offset);
            fz += __shfl_down_sync(mask, fz, offset);
        }

        if (lane_id == 0) {
            force[warp_id] = fx;
            force[num_nodes + warp_id] = fy;
            force[2 * num_nodes + warp_id] = fz;
        }
    }

    __global__ void calc_force_backward_kernel(
        float* __restrict__ grad_diff_E, 
        const float* __restrict__ grad_force, 
        const int64_t* __restrict__ dst_node_ptr, 
        const int64_t num_nodes, 
        const int64_t num_edges
    ) {
        const int64_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        const int64_t warp_id = tid / 32;
        const int lane_id = threadIdx.x % 32;
        if (warp_id >= num_nodes) return;

        const int64_t start_edge_idx = dst_node_ptr[warp_id];
        const int64_t end_edge_idx = dst_node_ptr[warp_id + 1];

        float gF_x = grad_force[warp_id];
        float gF_y = grad_force[num_nodes + warp_id];
        float gF_z = grad_force[2 * num_nodes + warp_id];
        for (int64_t e = start_edge_idx + lane_id; e < end_edge_idx; e += 32) {
            grad_diff_E[e] = gF_x;
            grad_diff_E[num_edges + e] = gF_y;
            grad_diff_E[2 * num_edges + e] = gF_z;
        }
    }
}

torch::Tensor calc_force_forward(
    torch::Tensor diff_E, 
    torch::Tensor dst_node_ptr
) {
    const int64_t num_edges = diff_E.size(1);
    const int64_t num_nodes = dst_node_ptr.size(0) - 1;
    auto force = torch::empty({3, num_nodes}, diff_E.options());

    int64_t num_threads = 256;
    int64_t num_warps = num_threads / 32;
    int64_t num_blocks = (num_nodes + num_warps - 1) / num_warps;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    calc_force_forward_kernel<<<num_blocks, num_threads, 0, stream>>>(
        force.data_ptr<float>(), 
        diff_E.data_ptr<float>(), 
        dst_node_ptr.data_ptr<int64_t>(), 
        num_nodes, 
        num_edges
    );

    return force;
}

torch::Tensor calc_force_backward(
    torch::Tensor diff_E, 
    torch::Tensor grad_force, 
    torch::Tensor dst_node_ptr
) {
    const int64_t num_nodes = dst_node_ptr.size(0) - 1;
    const int64_t num_edges = diff_E.size(1);

    auto grad_diff_E = torch::empty({3, num_edges}, diff_E.options());

    int64_t num_threads = 256;
    int64_t num_warps = num_threads / 32;
    int64_t num_blocks = (num_nodes + num_warps - 1) / num_warps;

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    calc_force_backward_kernel<<<num_blocks, num_threads, 0, stream>>>(
        grad_diff_E.data_ptr<float>(), 
        grad_force.data_ptr<float>(), 
        dst_node_ptr.data_ptr<int64_t>(), 
        num_nodes, 
        num_edges
    );

    return grad_diff_E;
}