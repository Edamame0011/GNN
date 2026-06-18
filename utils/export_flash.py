import torch
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader
import torch.nn as nn
from model.schnet_model_flash import SchNet_dict, GaussianRBF, InteractionBlock, ShiftedSoftplus, TypeEmbedding, CutoffFunction
from utils.preprocess import CustomData
import argparse
from torch.func import functional_call, grad
from torch.fx.experimental.proxy_tensor import make_fx
from utils.MyGraphFunction import fused_csr_cfconv, fused_distance_gaussian_rbf_cutoff, _fused_distance_gaussian_rbf_cutoff_fake, _fused_csr_cfconv_fake

#モデル
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
        self.register_buffer('centers', torch.linspace(0.0, cutoff, num_gaussians), persistent=False)
        spacing = cutoff / max(1, (num_gaussians - 1))
        self.gamma = -0.5 / (spacing * spacing)

    def forward(
            self, 
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_weight: torch.Tensor, 
            dst_ptr: torch.Tensor
        ):
        #埋め込み
        h = self.embedding(x) #(N, hidden_dim)

        i = edge_index[0]
        j = edge_index[1]
        
        distances = torch.norm(edge_weight, dim=0)

        rbf_expansion = fused_distance_gaussian_rbf_cutoff(
            distances, self.centers, self.gamma, self.cutoff
        )

        #Interactionレイヤー
        for interaction in self.interactions:
            h = interaction(h, rbf_expansion, distances, j, i, dst_ptr, self.cutoff)

        #各粒子のエネルギー
        energy = self.output(h) #(N, 1)
        total_energy = energy.sum()
        
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

        def forward(self, x, edge_index, edge_weight, dst_ptr):
            # 登録したParameterとBufferをタプルに束ねて渡す
            state_values = tuple(getattr(self, f"state_{i}") for i in range(self.num_states))
            return self.traced_graph(state_values, x, edge_index, edge_weight, dst_ptr)

def main():
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = "変換前モデルのパス"
    )
    parser.add_argument(
        '-o', '--output', type = str, 
        default = "./exported_model.pt2",
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
    def compute_energy_and_force(
            param_values_tuple, 
            x: torch.Tensor, 
            edge_index: torch.Tensor, 
            edge_weight: torch.Tensor, 
            dst_ptr: torch.Tensor
        ):
        p = {k: v for k, v in zip(state_keys, param_values_tuple)}

        def compute_energy(p_dict, x, edge_index, edge_weight, dst_ptr):
            y = functional_call(model, p_dict, (x, edge_index, edge_weight, dst_ptr))
            return y
        
        grad_fn = grad(compute_energy, argnums=3)

        total_energy = functional_call(model, p, (x, edge_index, edge_weight, dst_ptr))
        forces = -grad_fn(p, x, edge_index, edge_weight, dst_ptr)

        return total_energy, forces
    
    num_nodes = 50
    num_edges = 100

    # ダミー入力
    x = torch.randint(0, 100, (num_nodes, ), dtype=torch.long, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.int32, device=device)
    edge_weight = torch.randn((3, num_edges), dtype=torch.float, device=device)
    dst_ptr = torch.randint(0, num_edges, (num_nodes + 1,), dtype=torch.int32, device=device)
    dst_ptr, _ = torch.sort(dst_ptr)
    dst_ptr[0] = 0
    dst_ptr[-1] = num_edges

    edge_index = edge_index.to(torch.int32)
    dst_ptr = dst_ptr.to(torch.int32)

    # torch.funcの展開
    traced_graph = make_fx(compute_energy_and_force, tracing_mode="symbolic")(state_values, x, edge_index, edge_weight, dst_ptr)

    wrapper = ExportWrapper(traced_graph, state_keys, state_dict).to(device)
    wrapper.eval()

    # 要素数が可変な要素の設定
    dim_nodes = torch.export.Dim("num_nodes", min=1)
    dim_edges = torch.export.Dim("num_edges", min=1)
    
    dynamic_shapes = (
        {0: dim_nodes}, # x (num_nodes, )
        {1: dim_edges}, # pos (3, num_nodes)
        {1: dim_edges}, # edge_index (2, num_edges)
        {0: dim_nodes + 1}
    )

    # エクスポート
    exported_program = torch.export.export(
        wrapper, 
        args=(x, edge_index, edge_weight, dst_ptr), 
        dynamic_shapes=dynamic_shapes
    )
    torch._inductor.aoti_compile_and_package(
        exported_program, 
        package_path=output_path, 
        inductor_configs={
            "max_autotune": True, 
            "freezing": True
        }
    )
    print(f"モデルが正常に {output_path} へエクスポートされました。")

if __name__ == "__main__":
    main()