import torch
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader
import torch.nn as nn
from model.schnet_model import SchNet_dict, GaussianRBF, InteractionBlock, ShiftedSoftplus, TypeEmbedding, CutoffFunction
from utils.preprocess import CustomData
import argparse
from torch.func import functional_call, grad
from torch.fx.experimental.proxy_tensor import make_fx
import torch_tensorrt
import gc

# エネルギーのみを計算するモデル
class SchNetModel(nn.Module):
    def __init__(self, hidden_dim, num_gaussians, cutoff, num_filters, num_interactions, type_num = 100):
        super().__init__()

        self.setups = SchNet_dict(hidden_dim, num_gaussians, num_filters, num_interactions, cutoff, type_num)

        self.embedding = TypeEmbedding(type_num, hidden_dim)
        self.rbf = GaussianRBF(num_gaussians, cutoff)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_dim, num_gaussians, num_filters)
            self.interactions.append(block)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            ShiftedSoftplus(), 
            nn.Linear(hidden_dim // 2, 1)
        )

        self.cutoff = cutoff
        self.cutoff_function = CutoffFunction()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        #埋め込み
        h = self.embedding(x) #(N, hidden_dim)

        #RBF展開
        distances = torch.norm(edge_weight, dim=0)
        rbf_expansion = self.rbf(distances) # (num_edges, num_gaussians)

        #i: 送信元のノードのインデックス (num_edges, )
        #j: 送信先のノードのインデックス (num_edges, )
        i = edge_index[0]
        j = edge_index[1]      

        #原子間距離のカットオフ
        C = self.cutoff_function(distances, self.cutoff) #(num_edges, )

        #Interactionレイヤー
        for interaction in self.interactions:
            h = interaction(h, i, j, rbf_expansion, C)

        #各粒子のエネルギー
        energy = self.output(h) #(N, 1)

        total_energy = energy.sum() #全ノードの合計エネルギー
        
        return total_energy

class ExportWrapper(nn.Module):
        def __init__(self, traced_graph, state_keys, original_state_dict):
            super().__init__()
            self.traced_graph = traced_graph
            self.num_states = len(state_keys)

            # Parameter と Buffer を正しく判定してラッパーに再登録する
            # （リスト等に入れるとONNXエクスポート時に漏れるため、getattr/setattrを活用）
            for i, k in enumerate(state_keys):
                v = original_state_dict[k]
                if isinstance(v, nn.Parameter):
                    self.register_parameter(f"state_{i}", nn.Parameter(v.detach(), requires_grad=v.requires_grad))
                else:
                    self.register_buffer(f"state_{i}", v.detach())

        def forward(self, x, edge_index, edge_weight):
            # 登録したParameterとBufferをタプルに束ねて渡す
            state_values = tuple(getattr(self, f"state_{i}") for i in range(self.num_states))
            return self.traced_graph(state_values, x, edge_index, edge_weight)

def main():
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = "変換前モデルのパス"
    )
    parser.add_argument(
        '-o', '--output', type = str, 
        default = "./exported_model_trt.ts",
        help = "変換後モデルの保存先"
    )
    parser.add_argument(
        '--config', type = str, 
        default = "./configs/config.json", 
        help = "configファイルのパス"
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output
    config_path = args.config

    # configファイルの読み込み
    config = load_config(config_path)
    num_interactions    = config["num_interactions"]
    cutoff              = config["cutoff"]
    num_gaussians       = config["num_gaussians"]
    hidden_dim          = config["hidden_dim"]
    num_filters         = config["num_filters"]
    data_path           = config["data_path"]

    device = torch.device("cuda")

    # モデルの読み込み
    model = SchNetModel(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    state_dict = torch.load(input_path, weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # パラメータの名前と値を抽出
    param_dict = dict(model.named_parameters())
    buffer_dict = dict(model.named_buffers())
    
    # 2つを結合して1つの辞書にする
    state_dict = {**param_dict, **buffer_dict}
    state_keys = list(state_dict.keys())
    state_values = tuple(state_dict.values())   

    # 力とエネルギーを計算する純粋関数
    def compute_energy_and_force(param_values_tuple, x, edge_index, edge_weight):
        p = {k: v for k, v in zip(state_keys, param_values_tuple)}

        def compute_energy(p_dict, x, edge_index, edge_weight):
            y = functional_call(model, p_dict, (x, edge_index, edge_weight))
            return y
        
        grad_fn = grad(compute_energy, argnums=3)

        total_energy = functional_call(model, p, (x, edge_index, edge_weight))
        diff_E = grad_fn(p, x, edge_index, edge_weight)
        i, j = edge_index[0], edge_index[1]
        forces = x.new_zeros(3, x.size(0), dtype=edge_weight.dtype)
        forces.index_add_(1, i, diff_E)
        forces.index_add_(1, j, -diff_E)

        return total_energy, forces

    # ダミー入力
    num_nodes = 10
    num_edges = 128
    x = torch.randint(0, 100, (num_nodes, ), dtype=torch.long, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long, device=device)
    edge_weight = torch.randn(3, num_edges, dtype=torch.float, device=device)

    # torch.funcの展開
    traced_graph = make_fx(compute_energy_and_force, tracing_mode="symbolic")(state_values, x, edge_index, edge_weight)

    wrapper = ExportWrapper(traced_graph, state_keys, state_dict).to(device)
    wrapper.eval()

    # 要素数が可変な要素の設定
    num_edges = torch.export.Dim("num_edges", min=1)
    
    dynamic_shapes = (
        None, # data.x (num_nodes, )
        {1: num_edges}, # data.edge_index (2, num_edges)
        {1: num_edges}, # data.edge_weight (3, num_edges)
    )

    # エクスポート
    exported_program = torch.export.export(
        wrapper, 
        args=(x, edge_index, edge_weight), 
        dynamic_shapes=dynamic_shapes
    )

    del model, wrapper, traced_graph, state_dict, param_dict, buffer_dict
    gc.collect()
    torch.cuda.empty_cache()

    trt_compiled = torch_tensorrt.compile(
        exported_program, 
        ir="dynamo", 
        inputs=[
            torch_tensorrt.Input(
                shape=(1800, ), 
                dtype=torch.int64
            ), 
            torch_tensorrt.Input(
                min_shape=(2, 60000), 
                opt_shape=(2, 65000), 
                max_shape=(2, 70000)
            ), 
            torch_tensorrt.Input(
                min_shape=(3, 60000), 
                opt_shape=(3, 65000), 
                max_shape=(3, 70000)
            )
        ], 
        workspace_size=4 << 30
    )

    dummy_input1 = torch.randint(0, 2, (1800,)).long()       
    dummy_input2 = torch.randint(0, 10, (2, 65000)).long()                     
    dummy_input3 = torch.randn(3, 65000)                     

    torch_tensorrt.save(
            trt_compiled, 
            output_path, 
            output_format="torchscript", 
            inputs=(dummy_input1, dummy_input2, dummy_input3)
        )
    print(f"モデルが正常に {output_path} へエクスポートされました。")

if __name__ == "__main__":
    main()