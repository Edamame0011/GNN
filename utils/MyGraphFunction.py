import torch
import my_graph_ops

def message_aggregate(
        messages: torch.Tensor, 
        dst_node_ptr: torch.Tensor, 
        dst_list: torch.Tensor, 
        num_nodes: int
    ) -> torch.Tensor:
    # torch.ops 経由で呼び出す
    return torch.ops.my_graph_ops.message_aggregate(messages, dst_node_ptr, dst_list, num_nodes)

def calc_force(diff_E: torch.Tensor, dst_node_ptr: torch.Tensor) -> torch.Tensor:
    # torch.ops 経由で呼び出す
    return torch.ops.my_graph_ops.calc_force(diff_E, dst_node_ptr)