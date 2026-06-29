import torch
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader
import torch.nn as nn
from model.gnnff_energy_model import GNNFF_train
from model.gnnff_energy_model import SchNet_dict, GaussianRBF, InteractionBlock, ShiftedSoftplus, TypeEmbedding, CutoffFunction
from utils.preprocess import CustomData
import argparse
from torch.func import functional_call, grad
from torch.fx.experimental.proxy_tensor import make_fx

# energyモデル用のラッパー
class GNNFF_energy(nn.Module):
    def __init__(self, hidden_dim, num_gaussians, cutoff, num_filters, num_interactions, type_num = 100):
        super().__init__()

        self.setups = SchNet_dict(hidden_dim, num_gaussians, num_filters, num_interactions, cutoff, type_num)

        self.embedding = TypeEmbedding(type_num, hidden_dim)
        self.rbf = GaussianRBF(num_gaussians, cutoff)

        self.interactions = nn.ModuleList()
        for _ in range(num_interactions):
            block = InteractionBlock(hidden_dim, num_gaussians, num_filters)
            self.interactions.append(block)

        edge_hidden_dim = 2 * hidden_dim + num_gaussians

        self.output = nn.Sequential(
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2), 
            ShiftedSoftplus(), 
            nn.Linear(edge_hidden_dim // 2, 1)
        )

        self.calc_energy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), 
            ShiftedSoftplus(), 
            nn.Linear(hidden_dim // 2, 1)
        )

        self.cutoff = cutoff
        self.cutoff_function = CutoffFunction()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        edge_weight.requires_grad_(True)
            
        # 埋め込み
        h = self.embedding(x) #(N, hidden_dim)

        # RBF展開
        distances = torch.norm(edge_weight, dim=0)
        rbf_expansion = self.rbf(distances) # (num_edges, num_gaussians)

        i = edge_index[0]
        j = edge_index[1]

        # 原子間距離のカットオフ
        C = self.cutoff_function(distances, self.cutoff) #(num_edges, )

        # Interactionレイヤー
        for interaction in self.interactions:
            h = interaction(h, i, j, rbf_expansion, C)

        energy = self.calc_energy(h)
        total_energy = energy.sum() #全ノードの合計エネルギー

        return total_energy

def main():
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = "変換前モデルのパス"
    )
    parser.add_argument(
        '-o', '--output', type = str, 
        default = "./exported_model",
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
    model = GNNFF_train(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    state_dict = torch.load(input_path, weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    model_energy = GNNFF_energy(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    model_energy.load_state_dict(state_dict)
    model_energy.to(device)
    model_energy.eval()

    # ダミー入力
    data_list = torch.load(data_path, weights_only = False)
    loader = DataLoader(data_list, batch_size = 1, shuffle =  False)
    data = next(iter(loader))
    data = data.to(device)
    print(f"ダミー入力を{data_path}から読み込みました。")

    print("data.x shape:", data.x.shape)
    print("data.edge_index shape:", data.edge_index.shape)
    print("data.edge_weight shape:", data.edge_weight.shape)

    # 要素数が可変な要素の設定
    num_nodes = torch.export.Dim("num_nodes", min=1)
    num_edges = torch.export.Dim("num_edges", min=1)
    
    dynamic_shapes = (
        {0: num_nodes}, # data.x (num_nodes, )
        {1: num_edges}, # data.edge_index (2, num_edges)
        {1: num_edges}, # data.edge_weight (3, num_edges)
        None
    )

    # エクスポート
    exported_program = torch.export.export(
        model, 
        args=(data.x, data.edge_index, data.edge_weight, False), 
        dynamic_shapes=dynamic_shapes
    )
    torch._inductor.aoti_compile_and_package(
        exported_program, 
        package_path=f"{output_path}_force.pt2", 
        inductor_configs={
            "max_autotune": True, 
            "freezing": True
        }
    )
    print(f"Forceモデルが正常に {output_path}_force.pt2 へエクスポートされました。")

    # エクスポート
    exported_program = torch.export.export(
        model_energy, 
        args=(data.x, data.edge_index, data.edge_weight), 
        dynamic_shapes=(
            {0: num_nodes}, # data.x (num_nodes, )
            {1: num_edges}, # data.edge_index (2, num_edges)
            {1: num_edges}, # data.edge_weight (3, num_edges)
        )
    )
    torch._inductor.aoti_compile_and_package(
        exported_program, 
        package_path=f"{output_path}_energy.pt2", 
        inductor_configs={
            "max_autotune": True, 
            "freezing": True
        }
    )
    print(f"Energyモデルが正常に {output_path}_energy.pt2 へエクスポートされました。")

if __name__ == "__main__":
    main()