#include "kernels.hpp"

namespace {
    constexpr float PI = 3.14159265f;

    __global__ void fused_distance_gaussian_rbf_cutoff_kernel(
        const float* __restrict__ edge_weight_ptr,      // (num_edges)
        const float* __restrict__ centers_ptr,  // (num_rbf)
        float* __restrict__ rbf_output_ptr,     // (num_edges, num_rbf)
        const float cutoff, 
        const float gamma, 
        const int num_edges, 
        const int num_rbf
    ) {
        extern __shared__ float shared_centers[];
        for (int i = threadIdx.x; i < num_rbf; i += blockDim.x) {
            shared_centers[i] = centers_ptr[i];
        }
        __syncthreads();

        const int edge_idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (edge_idx >= num_edges) return;

        const float dist = edge_weight_ptr[edge_idx];
        const int base_offset = edge_idx * num_rbf;

        if (dist < cutoff) {
            const float cos_val = __cosf(dist * PI / cutoff);
            const float cutoff_val = 0.5f * (cos_val + 1.0f);

            for (int rbf_idx = 0; rbf_idx < num_rbf; rbf_idx ++) {
                float center = shared_centers[rbf_idx];
                float diff = dist - center;
                rbf_output_ptr[base_offset + rbf_idx] = expf(gamma * diff * diff) * cutoff_val;
            }
        } else {
            for (int rbf_idx = 0; rbf_idx < num_rbf; rbf_idx ++) {
                rbf_output_ptr[base_offset + rbf_idx] = 0.0f;
            }
        }
    }

    __global__ void fused_distance_gaussian_rbf_cutoff_grad_edge_weight_kernel(
        const float* __restrict__ edge_weight_ptr, 
        const float* __restrict__ centers_ptr, 
        const float* __restrict__ grad_rbf_ptr,
        float* __restrict__ grad_edge_weight_ptr, 
        float gamma, 
        float cutoff,
        int num_edges,
        int num_rbf
    ) {
        extern __shared__ float shared_centers[];
        for (int i = threadIdx.x; i < num_rbf; i += blockDim.x) {
            shared_centers[i] = centers_ptr[i];
        }
        __syncthreads();

        int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_idx >= num_edges) return;

        float dist = edge_weight_ptr[edge_idx];

        float grad_dist = 0.0f;

        if (dist < cutoff) {
            const float pi_over_cutoff = PI / cutoff;
            const float dist_pi_over_cutoff = dist * pi_over_cutoff;

            const float cos_val = __cosf(dist_pi_over_cutoff);
            const float cutoff_val = 0.5f * (cos_val + 1.0f);
            const float d_cutoff_val = -0.5f * pi_over_cutoff * __sinf(dist_pi_over_cutoff);

            const int base_offset = edge_idx * num_rbf;

            for (int rbf_idx = 0; rbf_idx < num_rbf; rbf_idx ++) {
                float center = shared_centers[rbf_idx];
                float diff = dist - center;

                float rbf_val = expf(gamma * diff * diff);
                float d_rbf_val = rbf_val * 2.0f * gamma * diff;

                float d_output = (d_rbf_val * cutoff_val) + (rbf_val * d_cutoff_val);

                float grad_out = grad_rbf_ptr[base_offset + rbf_idx];
                grad_dist += grad_out * d_output;
            }
        }

        grad_edge_weight_ptr[edge_idx] = grad_dist;
    }

    __global__ void fused_csr_cfconv_kernel(
        const float* x_ptr,             // (num_nodes, feat_dim)
        const float* filter_out_ptr,    // (num_edges, feat_dim)
        const float* edge_weight_ptr,   // (num_edges)
        const int32_t* edge_src_ptr,        // (num_edges)
        const int32_t* dst_ptr_ptr,         // (num_nodes + 1)
        float* output_ptr,              // (num_nodes, feat_dim)
        const float cutoff, 
        const int num_nodes, 
        const int feat_dim
    ) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        int num_warps = (gridDim.x * blockDim.x) / 32;

        for (int node_idx = warp_id; node_idx < num_nodes; node_idx += num_warps) {
            int seg_start, seg_end;

            if (lane_id == 0) {
                seg_start = dst_ptr_ptr[node_idx];
                seg_end = dst_ptr_ptr[node_idx + 1];
            }

            seg_start = __shfl_sync(0xffffffff, seg_start, 0);
            seg_end = __shfl_sync(0xffffffff, seg_end, 0);

            for (int base_f = 0; base_f < feat_dim; base_f += 32) {
                int f = base_f + lane_id;
                unsigned int active_mask = __ballot_sync(0xffffffff, f < feat_dim);

                if (f < feat_dim) {
                    float acc = 0.0f;

                    for (int e_csr = seg_start; e_csr < seg_end; e_csr ++) {
                        float dist = edge_weight_ptr[e_csr];

                        if (dist < cutoff) {
                            float C;
                            if (lane_id == 0) {
                                C = 0.5f * (__cosf(dist * PI / cutoff) + 1.0f);
                            }
                            C = __shfl_sync(0xffffffff, C, 0);

                            int src_node = edge_src_ptr[e_csr];

                            float filter_val = filter_out_ptr[e_csr * feat_dim + f];
                            float x_j = x_ptr[src_node * feat_dim + f];

                            acc += x_j * filter_val * C;
                        }
                    }

                    output_ptr[node_idx * feat_dim + f] = acc;
                }
            }
        }
    }

    __global__ void fused_csr_grad_x_kernel(
        const float* __restrict__ grad_output_ptr, 
        const float* __restrict__ filter_out_ptr, 
        const float* __restrict__ edge_weight_ptr, 
        const int32_t* __restrict__ edge_src_ptr, 
        const int32_t* __restrict__ edge_dst_ptr, 
        float* __restrict__ grad_x_ptr, 
        const float cutoff, 
        const int num_edges, 
        const int feat_dim
    ) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int warp_id = tid / 32;
        int lane_id = tid % 32;

        if (warp_id >= num_edges) return;

        int src_node, dst_node;
        float dist;

        if (lane_id == 0) {
            src_node = edge_src_ptr[warp_id];
            dst_node = edge_dst_ptr[warp_id];
            dist = edge_weight_ptr[warp_id];
        }

        src_node = __shfl_sync(0xffffffff, src_node, 0);
        dst_node = __shfl_sync(0xffffffff, dst_node, 0);
        dist = __shfl_sync(0xffffffff, dist, 0);

        if (dist >= cutoff) return;
        
        float C;
        if (lane_id == 0) {
            C = 0.5f * (__cosf(dist * PI / cutoff) + 1.0f);
        }
        C = __shfl_sync(0xffffffff, C, 0);

        const int filter_offset = warp_id * feat_dim;
        const int grad_out_offset = dst_node * feat_dim;
        const int grad_x_offset = src_node * feat_dim;

        for (int f = lane_id; f < feat_dim; f += 32) {
            float filter_val = filter_out_ptr[filter_offset + f];
            float W = filter_val * C;
            float grad_out_val = grad_output_ptr[grad_out_offset + f];
            float msg = grad_out_val * W;

            atomicAdd(&grad_x_ptr[grad_x_offset + f], msg);
        }
    }

    __global__ void fused_grad_filter_out_kernel(
        const float* __restrict__ x_ptr,             // (num_nodes, feat_dim)
        const float* __restrict__ grad_output_ptr,   // (num_edges, feat_dim)
        const float* __restrict__ edge_weight_ptr,   // (num_edges)
        const int32_t* __restrict__ edge_src_ptr,        // (num_edges)
        const int32_t* __restrict__ edge_dst_ptr,        // (num_edges)
        float* __restrict__ grad_filter_out_ptr,     // (num_edges, feat_dim)
        const float cutoff, 
        const int num_edges, 
        const int feat_dim
    ) {
        size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
        size_t total_elements = num_edges * feat_dim;

        if (idx >= total_elements) return;
        
        size_t edge_idx = idx / feat_dim;
        size_t f = idx % feat_dim;

        float dist = edge_weight_ptr[edge_idx];
        float C = 0.0f;

        if (dist < cutoff) {
            C = 0.5f * (__cosf(dist * PI / cutoff) + 1.0f);
        }

        int src_node = edge_src_ptr[edge_idx];
        int dst_node = edge_dst_ptr[edge_idx];

        float x_j = x_ptr[src_node * feat_dim + f];
        float grad_j = grad_output_ptr[dst_node * feat_dim + f];

        float grad_filter = x_j * grad_j * C;

        grad_filter_out_ptr[idx] = grad_filter;
    }

    __global__ void fused_grad_edge_weight_kernel(
        const float* grad_output_ptr,   // (num_nodes, feat_dim)
        const float* x_ptr,             // (num_nodes, feat_dim)
        const float* filter_out_ptr,    // (num_edges, feat_dim)
        const float* edge_weight_ptr,   // (num_edges)          
        const int32_t* edge_src_ptr,    // (num_edges)          
        const int32_t* dst_ptr_ptr,     // (num_nodes + 1)      
        float* grad_edge_weight_ptr,    // (num_edges)          
        const float cutoff, 
        const int num_nodes, 
        const int feat_dim
    ) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        int num_warps = (gridDim.x * blockDim.x) / 32;

        for (int node_idx = warp_id; node_idx < num_nodes; node_idx += num_warps) {
            int seg_start, seg_end;

            if (lane_id == 0) {
                seg_start = dst_ptr_ptr[node_idx];
                seg_end = dst_ptr_ptr[node_idx + 1];
            }
            seg_start = __shfl_sync(0xffffffff, seg_start, 0);
            seg_end = __shfl_sync(0xffffffff, seg_end, 0);

            for (int e_csr = seg_start; e_csr < seg_end; e_csr++) {
                float dist;
                if (lane_id == 0) {
                    dist = edge_weight_ptr[e_csr];
                }
                dist = __shfl_sync(0xffffffff, dist, 0);

                if (dist < cutoff) {
                    float dC_ddist;
                    int src_node;

                    if (lane_id == 0) {
                        dC_ddist = -0.5f * (PI / cutoff) * __sinf(dist * PI / cutoff);
                        src_node = edge_src_ptr[e_csr];
                    }
                    dC_ddist = __shfl_sync(0xffffffff, dC_ddist, 0);
                    src_node = __shfl_sync(0xffffffff, src_node, 0);

                    float sum_f = 0.0f;

                    for (int base_f = 0; base_f < feat_dim; base_f += 32) {
                        int f = base_f + lane_id;

                        if (f < feat_dim) {
                            float grad_out = grad_output_ptr[node_idx * feat_dim + f];
                            float filter_val = filter_out_ptr[e_csr * feat_dim + f];
                            float x_j = x_ptr[src_node * feat_dim + f];

                            sum_f += grad_out * x_j * filter_val;
                        }
                    }

                    for (int offset = 16; offset > 0; offset /= 2) {
                        sum_f += __shfl_down_sync(0xffffffff, sum_f, offset);
                    }

                    if (lane_id == 0) {
                        grad_edge_weight_ptr[e_csr] = sum_f * dC_ddist;
                    }
                } else {
                    if (lane_id == 0) {
                        grad_edge_weight_ptr[e_csr] = 0.0f;
                    }
                }
            }
        }
    }
}

namespace FlashSchNet::kernels {
    torch::Tensor fused_distance_gaussian_rbf_cutoff(
        const torch::Tensor& edge_weight, 
        const torch::Tensor& centers, 
        float gamma, 
        float cutoff
    ) {
        int num_edges = edge_weight.size(0);
        int num_rbf = centers.size(0);

        auto rbf_output = torch::empty({num_edges, num_rbf}, edge_weight.options());

        if (num_edges == 0) return rbf_output;

        int num_threads = 256;
        int num_blocks = (num_threads + num_edges - 1) / num_threads;
        size_t shared_mem_size = num_rbf * sizeof(float);

        fused_distance_gaussian_rbf_cutoff_kernel<<<num_blocks, num_threads, shared_mem_size>>>(
            edge_weight.data_ptr<float>(), 
            centers.data_ptr<float>(), 
            rbf_output.data_ptr<float>(), 
            cutoff, 
            gamma, 
            num_edges, 
            num_rbf
        );

        return rbf_output;
    }

    torch::Tensor fused_distance_gaussian_rbf_cutoff_grad_pos(
        const torch::Tensor& edge_weight,
        const torch::Tensor& centers,
        const torch::Tensor& grad_rbf, 
        float gamma,
        float cutoff
    ) {
        int num_edges = edge_weight.size(0);
        int num_rbf = centers.size(0);

        auto grad_edge_weight = torch::empty({num_edges}, edge_weight.options());

        int num_threads = 256;
        int num_blocks = (num_threads + num_edges - 1) / num_threads;
        size_t shared_mem_size = num_rbf * sizeof(float);

        fused_distance_gaussian_rbf_cutoff_grad_edge_weight_kernel<<<num_blocks, num_threads, shared_mem_size>>>(
            edge_weight.data_ptr<float>(), 
            centers.data_ptr<float>(), 
            grad_rbf.data_ptr<float>(), 
            grad_edge_weight.data_ptr<float>(), 
            gamma, 
            cutoff, 
            num_edges, 
            num_rbf
        );

        return grad_edge_weight;
    }

    torch::Tensor fused_csr_cfconv(
        const torch::Tensor& x, 
        const torch::Tensor& filter_out, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& dst_ptr, 
        float cutoff
    ) {
        int num_nodes = x.size(0);
        int feat_dim = x.size(1);

        auto output = torch::zeros({num_nodes, feat_dim}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

        int num_threads = 256;
        int num_warps = num_threads / 32;
        int num_blocks = (num_warps + num_nodes - 1) / num_warps;

        fused_csr_cfconv_kernel<<<num_blocks, num_threads>>>(
            x.data_ptr<float>(), 
            filter_out.data_ptr<float>(), 
            edge_weight.data_ptr<float>(), 
            edge_src.data_ptr<int32_t>(), 
            dst_ptr.data_ptr<int32_t>(), 
            output.data_ptr<float>(), 
            cutoff, 
            num_nodes, 
            feat_dim
        );

        return output;
    }

    torch::Tensor fused_csr_grad_x(
        const torch::Tensor& grad_output, 
        const torch::Tensor& filter_out, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& edge_dst, 
        float cutoff
    ) {
        int num_nodes = grad_output.size(0);
        int feat_dim = grad_output.size(1);
        int num_edges = edge_src.size(0);
        auto grad_x = torch::zeros({num_nodes, feat_dim}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

        int num_threads = 256;
        int num_warps = num_threads / 32;
        int num_blocks = (num_warps + num_edges - 1) / num_warps;

        fused_csr_grad_x_kernel<<<num_blocks, num_threads>>>(
            grad_output.data_ptr<float>(), 
            filter_out.data_ptr<float>(), 
            edge_weight.data_ptr<float>(), 
            edge_src.data_ptr<int32_t>(), 
            edge_dst.data_ptr<int32_t>(), 
            grad_x.data_ptr<float>(), 
            cutoff, 
            num_edges, 
            feat_dim
        );

        return grad_x;
    }

    torch::Tensor fused_grad_filter_out(
        const torch::Tensor& x, 
        const torch::Tensor& grad_output, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& edge_dst, 
        float cutoff
    ) {
        int feat_dim = x.size(1);
        int num_edges = edge_src.size(0);
        
        auto grad_filter_out = torch::empty({num_edges, feat_dim}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

        int num_threads = 256;
        int total_elements = num_edges * feat_dim;
        int num_blocks = (num_threads + total_elements - 1) / num_threads;

        fused_grad_filter_out_kernel<<<num_blocks, num_threads>>>(
            x.data_ptr<float>(), 
            grad_output.data_ptr<float>(), 
            edge_weight.data_ptr<float>(), 
            edge_src.data_ptr<int32_t>(), 
            edge_dst.data_ptr<int32_t>(), 
            grad_filter_out.data_ptr<float>(), 
            cutoff, 
            num_edges, 
            feat_dim
        );

        return grad_filter_out;
    }

    torch::Tensor fused_grad_edge_weight(
        const torch::Tensor& grad_output, 
        const torch::Tensor& x, 
        const torch::Tensor& filter_out, 
        const torch::Tensor& edge_weight, 
        const torch::Tensor& edge_src, 
        const torch::Tensor& dst_ptr, 
        float cutoff
    ) {
        int num_nodes = x.size(0);
        int feat_dim = x.size(1);

        auto grad_edge_weight = torch::empty_like(edge_weight);

        int num_threads = 256;
        int num_warps = num_threads / 32;
        int num_blocks = (num_warps + num_nodes - 1) / num_warps;

        fused_grad_edge_weight_kernel<<<num_blocks, num_threads>>>(
            grad_output.data_ptr<float>(), 
            x.data_ptr<float>(), 
            filter_out.data_ptr<float>(), 
            edge_weight.data_ptr<float>(), 
            edge_src.data_ptr<int32_t>(), 
            dst_ptr.data_ptr<int32_t>(), 
            grad_edge_weight.data_ptr<float>(), 
            cutoff, 
            num_nodes, 
            feat_dim
        );

        return grad_edge_weight;
    }
}