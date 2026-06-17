#include "kernels.hpp"

namespace {
    constexpr float PI = 3.14159265f;

    __global__ void fused_distance_gaussian_rbf_cutoff_kernel(
        const float* __restrict__ pos_ptr,      // (3, num_nodes)
        const int64_t* __restrict__ edge_src_ptr,   // (num_edges)
        const int64_t* __restrict__ edge_dst_ptr,   // (num_edges)
        const float* __restrict__ centers_ptr,  // (num_rbf)
        float* __restrict__ dist_output_ptr,    // (num_edges)
        float* __restrict__ rbf_output_ptr,     // (num_edges, num_rbf)
        const float cutoff, 
        const float gamma, 
        const int num_edges, 
        const int num_rbf, 
        const int num_nodes
    ) {
        int edge_idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (edge_idx >= num_edges) return;

        int src_node = edge_src_ptr[edge_idx];
        int dst_node = edge_dst_ptr[edge_idx];

        const float src_x = pos_ptr[0 * num_nodes + src_node];
        const float src_y = pos_ptr[1 * num_nodes + src_node];
        const float src_z = pos_ptr[2 * num_nodes + src_node];

        const float dst_x = pos_ptr[0 * num_nodes + dst_node];
        const float dst_y = pos_ptr[1 * num_nodes + dst_node];
        const float dst_z = pos_ptr[2 * num_nodes + dst_node];

        const float dx = dst_x - src_x;
        const float dy = dst_y - src_y;
        const float dz = dst_z - src_z;

        const float dist = sqrtf(dx * dx + dy * dy + dz * dz);

        dist_output_ptr[edge_idx] = dist;

        float cutoff_val = 0.0f;
        if (dist < cutoff) {
            float cos_val = __cosf(dist * PI / cutoff);
            cutoff_val = 0.5f * (cos_val + 1.0f); 
        }

        int base_offset = edge_idx * num_rbf;

        for (int rbf_idx = 0; rbf_idx < num_rbf; rbf_idx ++) {
            float center = centers_ptr[rbf_idx];
            float diff = dist - center;
            float rbf_val = expf(gamma * diff * diff) * cutoff_val;

            rbf_output_ptr[base_offset + rbf_idx] = rbf_val;
        }
    }

    __global__ void fused_distance_gaussian_rbf_cutoff_grad_pos_kernel(
        const float* __restrict__ pos,
        const int64_t* __restrict__ edge_src,
        const int64_t* __restrict__ edge_dst,
        const float* __restrict__ centers,
        const float* __restrict__ distances,
        const float* __restrict__ grad_distances,
        const float* __restrict__ grad_rbf,
        float* __restrict__ grad_pos,
        float gamma,
        float cutoff_upper,
        int num_edges,
        int num_rbf, 
        int num_nodes
    ) {
        int e = blockIdx.x * blockDim.x + threadIdx.x;
        if (e >= num_edges) return;

        float dist = distances[e];

        if (dist >= cutoff_upper) return; 

        float dist_safe = fmaxf(dist, 1e-8f);

        int src = edge_src[e];
        int dst = edge_dst[e];

        float dx = pos[0 * num_nodes + dst] - pos[0 * num_nodes + src];
        float dy = pos[1 * num_nodes + dst] - pos[1 * num_nodes + src];
        float dz = pos[2 * num_nodes + dst] - pos[2 * num_nodes + src];

        float dir_x = dx / dist_safe;
        float dir_y = dy / dist_safe;
        float dir_z = dz / dist_safe;

        float pi_over_c = PI / cutoff_upper;
        float cos_val = __cosf(dist * pi_over_c);
        float sin_val = __sinf(dist * pi_over_c);

        float cutoff_val = 0.5f * (cos_val + 1.0f);
        float d_cutoff_d_dist = -0.5f * pi_over_c * sin_val;

        float grad_dist_from_rbf = 0.0f;
        
        if (grad_rbf != nullptr) {
            for (int k = 0; k < num_rbf; ++k) {
                float center = centers[k];
                float diff = dist - center;
                float exp_term = expf(gamma * diff * diff);

                // Chain rule
                float d_rbf_d_dist = 2.0f * gamma * diff * exp_term * cutoff_val + exp_term * d_cutoff_d_dist;

                float g_rbf = grad_rbf[e * num_rbf + k];
                grad_dist_from_rbf += g_rbf * d_rbf_d_dist;
            }
        }

        float total_grad_dist = grad_dist_from_rbf;
        if (grad_distances != nullptr) {
            total_grad_dist += grad_distances[e];
        }

        float grad_dr_x = total_grad_dist * dir_x;
        float grad_dr_y = total_grad_dist * dir_y;
        float grad_dr_z = total_grad_dist * dir_z;

        atomicAdd(&grad_pos[0 * num_nodes + dst], grad_dr_x);
        atomicAdd(&grad_pos[1 * num_nodes + dst], grad_dr_y);
        atomicAdd(&grad_pos[2 * num_nodes + dst], grad_dr_z);

        atomicAdd(&grad_pos[0 * num_nodes + src], -grad_dr_x);
        atomicAdd(&grad_pos[1 * num_nodes + src], -grad_dr_y);
        atomicAdd(&grad_pos[2 * num_nodes + src], -grad_dr_z);
    }

    __global__ void fused_csr_cfconv_kernel(
        const float* x_ptr,             // (num_nodes, feat_dim)
        const float* filter_out_ptr,    // (num_edges, feat_dim)
        const float* edge_weight_ptr,   // (num_edges)
        const int64_t* edge_src_ptr,        // (num_edges)
        const int64_t* csr_perm_ptr,        // (num_edges)
        const int64_t* dst_ptr_ptr,         // (num_nodes + 1)
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
            int seg_start = dst_ptr_ptr[node_idx];
            int seg_end = dst_ptr_ptr[node_idx + 1];

            for (int f = lane_id; f < feat_dim; f += 32) {
                float acc = 0.0f;

                for (int e_csr = seg_start; e_csr < seg_end; e_csr ++) {
                    int edge_idx = csr_perm_ptr[e_csr];
                    float dist = edge_weight_ptr[edge_idx];

                    if (dist < cutoff) {
                        float C = 0.5f * (__cosf(dist * PI / cutoff) + 1.0f);
                        int src_node = edge_src_ptr[edge_idx];

                        float filter_val = filter_out_ptr[edge_idx * feat_dim + f];
                        float x_j = x_ptr[src_node * feat_dim + f];

                        acc += x_j * filter_val * C;
                    }
                }

                output_ptr[node_idx * feat_dim + f] = acc;
            }
        }
    }

    __global__ void fused_src_csr_grad_x_kernel(
        const float* __restrict__ grad_output_ptr, 
        const float* __restrict__ filter_out_ptr, 
        const float* __restrict__ edge_weight_ptr, 
        const int64_t* __restrict__ edge_dst_ptr, 
        const int64_t* __restrict__ src_perm_ptr, 
        const int64_t* __restrict__ src_ptr_ptr, 
        float* __restrict__ grad_x_ptr, 
        const float cutoff, 
        const int num_nodes, 
        const int feat_dim
    ) {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        int warp_id = tid / 32;
        int lane_id = tid % 32;

        if (warp_id >= num_nodes) return;

        int seg_start = src_ptr_ptr[warp_id];
        int seg_end = src_ptr_ptr[warp_id + 1];

        for (int f = lane_id; f < feat_dim; f += 32) {
            float acc = 0.0f;

            for (int e_csr = seg_start; e_csr < seg_end; e_csr ++) {
                int edge_idx = src_perm_ptr[e_csr];
                int dst_node = edge_dst_ptr[edge_idx];
                float dist = edge_weight_ptr[edge_idx];

                if (dist < cutoff) {
                    float C = 0.5f * (__cosf(dist * PI / cutoff) + 1.0f);

                    float filter_val = filter_out_ptr[edge_idx * feat_dim + f];
                    float W = filter_val * C;
                    float grad_dst = grad_output_ptr[dst_node * feat_dim + f];

                    acc += grad_dst * W;
                }
            }

            grad_x_ptr[warp_id * feat_dim + f] = acc;
        }
    }

    __global__ void fused_grad_filter_out_kernel(
        const float* __restrict__ x_ptr,             // (num_nodes, feat_dim)
        const float* __restrict__ grad_output_ptr,   // (num_edges, feat_dim)
        const float* __restrict__ edge_weight_ptr,   // (num_edges)
        const int64_t* __restrict__ edge_src_ptr,        // (num_edges)
        const int64_t* __restrict__ edge_dst_ptr,        // (num_edges)
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
}

namespace FlashSchNet::kernels {
    std::tuple<torch::Tensor, torch::Tensor> fused_distance_gaussian_rbf_cutoff(
        torch::Tensor pos, 
        torch::Tensor edge_src, 
        torch::Tensor edge_dst, 
        torch::Tensor centers, 
        float gamma, 
        float cutoff
    ) {
        int num_edges = edge_src.size(0);
        int num_rbf = centers.size(0);
        int num_nodes = pos.size(1);

        auto opt = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
        auto distances = torch::empty({num_edges}, opt);
        auto rbf_output = torch::empty({num_edges, num_rbf}, opt);

        if (num_edges == 0) return std::make_tuple(distances, rbf_output);

        int num_threads = 256;
        int num_blocks = (num_threads + num_edges - 1) / num_threads;

        fused_distance_gaussian_rbf_cutoff_kernel<<<num_blocks, num_threads>>>(
            pos.data_ptr<float>(), 
            edge_src.data_ptr<int64_t>(), 
            edge_dst.data_ptr<int64_t>(), 
            centers.data_ptr<float>(), 
            distances.data_ptr<float>(), 
            rbf_output.data_ptr<float>(), 
            cutoff, 
            gamma, 
            num_edges, 
            num_rbf, 
            num_nodes
        );

        return std::make_tuple(distances, rbf_output);
    }

    void fused_distance_gaussian_rbf_cutoff_grad_pos(
        torch::Tensor pos,
        torch::Tensor edge_src,
        torch::Tensor edge_dst,
        torch::Tensor centers,
        torch::Tensor distances,
        torch::Tensor grad_distances,
        torch::Tensor grad_rbf,
        torch::Tensor grad_pos,
        float gamma,
        float cutoff
    ) {
        int num_edges = edge_src.size(0);
        int num_rbf = centers.size(0);
        int num_nodes = pos.size(1);

        int num_threads = 256;
        int num_blocks = (num_threads + num_edges - 1) / num_threads;

        fused_distance_gaussian_rbf_cutoff_grad_pos_kernel<<<num_blocks, num_threads>>>(
            pos.data_ptr<float>(), 
            edge_src.data_ptr<int64_t>(), 
            edge_dst.data_ptr<int64_t>(), 
            centers.data_ptr<float>(), 
            distances.data_ptr<float>(), 
            grad_distances.data_ptr<float>(), 
            grad_rbf.data_ptr<float>(), 
            grad_pos.data_ptr<float>(), 
            gamma, 
            cutoff, 
            num_edges, 
            num_rbf, 
            num_nodes
        );
    }

    torch::Tensor fused_csr_cfconv(
        torch::Tensor x, 
        torch::Tensor filter_out, 
        torch::Tensor edge_weight, 
        torch::Tensor edge_src, 
        torch::Tensor dst_ptr, 
        torch::Tensor csr_perm, 
        int num_nodes, 
        float cutoff
    ) {
        int feat_dim = x.size(1);

        auto output = torch::zeros({num_nodes, feat_dim}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

        int num_threads = 256;
        int num_warps = num_threads / 32;
        int num_blocks = (num_warps + num_nodes - 1) / num_warps;

        fused_csr_cfconv_kernel<<<num_blocks, num_threads>>>(
            x.data_ptr<float>(), 
            filter_out.data_ptr<float>(), 
            edge_weight.data_ptr<float>(), 
            edge_src.data_ptr<int64_t>(), 
            csr_perm.data_ptr<int64_t>(), 
            dst_ptr.data_ptr<int64_t>(), 
            output.data_ptr<float>(), 
            cutoff, 
            num_nodes, 
            feat_dim
        );

        return output;
    }

    torch::Tensor fused_src_csr_grad_x(
        torch::Tensor grad_output, 
        torch::Tensor filter_out, 
        torch::Tensor edge_weight, 
        torch::Tensor edge_dst, 
        torch::Tensor src_ptr, 
        torch::Tensor src_perm, 
        int num_nodes, 
        float cutoff
    ) {
        int feat_dim = grad_output.size(1);
        auto grad_x = torch::zeros({num_nodes, feat_dim}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32));

        int num_threads = 256;
        int num_warps = num_threads / 32;
        int num_blocks = (num_warps + num_nodes - 1) / num_warps;

        fused_src_csr_grad_x_kernel<<<num_blocks, num_threads>>>(
            grad_output.data_ptr<float>(), 
            filter_out.data_ptr<float>(), 
            edge_weight.data_ptr<float>(), 
            edge_dst.data_ptr<int64_t>(), 
            src_perm.data_ptr<int64_t>(), 
            src_ptr.data_ptr<int64_t>(), 
            grad_x.data_ptr<float>(), 
            cutoff, 
            num_nodes, 
            feat_dim
        );

        return grad_x;
    }

    torch::Tensor fused_grad_filter_out(
        torch::Tensor x, 
        torch::Tensor grad_output, 
        torch::Tensor edge_weight, 
        torch::Tensor edge_src, 
        torch::Tensor edge_dst, 
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
            edge_src.data_ptr<int64_t>(), 
            edge_dst.data_ptr<int64_t>(), 
            grad_filter_out.data_ptr<float>(), 
            cutoff, 
            num_edges, 
            feat_dim
        );

        return grad_filter_out;
    }
}