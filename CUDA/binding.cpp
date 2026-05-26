#include <torch/extension.h>
#include <torch/script.h>

torch::Tensor calc_force_forward(torch::Tensor diff_E, torch::Tensor dst_node_ptr);
torch::Tensor calc_force_backward(torch::Tensor diff_E, torch::Tensor grad_force, torch::Tensor dst_node_ptr);
torch::Tensor message_agg_forward(torch::Tensor messages, torch::Tensor dst_node_ptr, int64_t num_nodes);
torch::Tensor message_agg_backward(torch::Tensor grad_agg_messages, torch::Tensor dst_list, torch::Tensor offsets, int64_t num_nodes);

class CalcForceFunction : public torch::autograd::Function<CalcForceFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx, 
        torch::Tensor diff_E, 
        torch::Tensor dst_node_ptr) 
    {
        ctx->save_for_backward({diff_E, dst_node_ptr});
        return calc_force_forward(diff_E, dst_node_ptr);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx, 
        torch::autograd::variable_list grad_outputs) 
    {
        auto saved = ctx->get_saved_variables();
        auto diff_E = saved[0];
        auto dst_node_ptr = saved[1];
        auto grad_force = grad_outputs[0];

        auto grad_diff_E = calc_force_backward(diff_E, grad_force, dst_node_ptr);
        
        // forwardの引数に対応する勾配を返す (dst_node_ptr は勾配不要なので未定義Tensor)
        return {grad_diff_E, torch::Tensor()};
    }
};

class MessageAggregateFunction : public torch::autograd::Function<MessageAggregateFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx, 
        torch::Tensor messages, 
        torch::Tensor offsets, 
        torch::Tensor dst_list, 
        int64_t num_nodes) 
    {
        ctx->save_for_backward({dst_list, offsets});
        ctx->saved_data["num_nodes"] = num_nodes;
        return message_agg_forward(messages, offsets, num_nodes);
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext *ctx, 
        torch::autograd::variable_list grad_outputs) 
    {
        auto saved = ctx->get_saved_variables();
        auto dst_list = saved[0];
        auto offsets = saved[1];
        auto grad_agg_messages = grad_outputs[0];

        int64_t num_nodes = ctx->saved_data["num_nodes"].toInt();

        auto grad_messages = message_agg_backward(grad_agg_messages, dst_list, offsets, num_nodes);

        return {grad_messages, torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor calc_force_op(torch::Tensor diff_E, torch::Tensor dst_node_ptr) {
    return CalcForceFunction::apply(diff_E, dst_node_ptr);
}

torch::Tensor message_aggregate_op(
    torch::Tensor messages, 
    torch::Tensor offsets, 
    torch::Tensor dst_list, 
    int64_t num_nodes) 
{
    return MessageAggregateFunction::apply(messages, offsets, dst_list, num_nodes);
}

// 関数の「型（スキーマ）」だけを定義するブロック
TORCH_LIBRARY(my_graph_ops, m) {
    m.def("calc_force(Tensor diff_E, Tensor dst_node_ptr) -> Tensor");
    m.def("message_aggregate(Tensor messages, Tensor dst_node_ptr, Tensor dst_list, int num_nodes) -> Tensor");
}

// 推論時 (CUDA実行時) の実体を登録するブロック
TORCH_LIBRARY_IMPL(my_graph_ops, CUDA, m) {
    m.impl("calc_force", &calc_force_op);
    m.impl("message_aggregate", &message_aggregate_op);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Custom Graph Operations for SchNet";
}