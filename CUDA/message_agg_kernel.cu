#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

namespace {
    // agg_messages[i]に、iに繋がっているすべてのjについてのmessages[j]を足し合わせる
    __global__  void message_agg_forward_kernel(
        float* __restrict__ agg_messages, 
        const float* __restrict__ messages, 
        const int64_t* __restrict__ dst_node_ptr, 
        const int64_t num_nodes, 
        const int64_t num_edges, 
        const int64_t num_feat
    ) {
        const int lane_id = threadIdx.x;
        const int64_t node_idx = threadIdx.y + blockIdx.y * blockDim.y;
        const int64_t feat_idx = blockIdx.x;
        if (node_idx >= num_nodes || feat_idx >= num_feat) return;

        const int64_t start_edge_idx = dst_node_ptr[node_idx];
        const int64_t end_edge_idx = dst_node_ptr[node_idx + 1];

        float m = 0.0f;
        for (int64_t e = start_edge_idx + lane_id; e < end_edge_idx; e += 32) {
            m += messages[feat_idx * num_edges + e];
        }

        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            m += __shfl_down_sync(mask, m, offset);
        }

        if (lane_id == 0) {
            agg_messages[node_idx * num_feat + feat_idx] = m;
        }
    }

    __global__ void message_agg_backward_kernel(
        float* __restrict__ grad_messages, 
        const float* __restrict__ grad_agg_messages, 
        const int64_t* __restrict__ dst_list, 
        const int64_t num_edges, 
        const int64_t num_feat
    ) {
        const int64_t feat_idx = threadIdx.x + blockIdx.x * blockDim.x;
        const int64_t edge_idx = threadIdx.y + blockIdx.y * blockDim.y;
        if (edge_idx >= num_edges || feat_idx >= num_feat) return;

        const int64_t i = dst_list[edge_idx];
        grad_messages[feat_idx * num_edges + edge_idx] = grad_agg_messages[i * num_feat + feat_idx];
    }
}

torch::Tensor message_agg_forward(
    torch::Tensor messages, 
    torch::Tensor dst_node_ptr, 
    int64_t num_nodes
) {
    const int64_t num_feat = messages.size(0);
    const int64_t num_edges = messages.size(1);
    auto agg_messages = torch::zeros({num_nodes, num_feat}, messages.options());

    dim3 num_threads(32, 8);
    dim3 num_blocks(
        num_feat, 
        (num_nodes + num_threads.y - 1) / num_threads.y
    );

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    message_agg_forward_kernel<<<num_blocks, num_threads, 0, stream>>>(
        agg_messages.data_ptr<float>(), 
        messages.data_ptr<float>(), 
        dst_node_ptr.data_ptr<int64_t>(), 
        num_nodes, 
        num_edges, 
        num_feat
    );

    return agg_messages;
}

torch::Tensor message_agg_backward(
    torch::Tensor grad_agg_messages, 
    torch::Tensor dst_list
) {
    const int64_t num_edges = dst_list.size(0);
    const int64_t num_feat = grad_agg_messages.size(1);
    auto grad_messages = torch::empty({num_feat, num_edges}, grad_agg_messages.options());

    dim3 num_threads(32, 8);
    dim3 num_blocks(
        (num_feat + num_threads.x - 1) / num_threads.x, 
        (num_edges + num_threads.y - 1) / num_threads.y
    );

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    message_agg_backward_kernel<<<num_blocks, num_threads, 0, stream>>>(
        grad_messages.data_ptr<float>(), 
        grad_agg_messages.data_ptr<float>(), 
        dst_list.data_ptr<int64_t>(), 
        num_edges, 
        num_feat
    );

    return grad_messages;
}