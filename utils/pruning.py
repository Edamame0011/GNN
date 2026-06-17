import torch
import torch.nn as nn
import argparse
from train.train_schnet import load_config
import torch_pruning as tp
from torch_geometric.loader import DataLoader
from utils.preprocess import CustomData
from model.schnet_model import SchNet_dict, GaussianRBF, InteractionBlock, ShiftedSoftplus, TypeEmbedding, CutoffFunction

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

def main():
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = "変換前モデルのパス"
    )
    parser.add_argument(
        '-o', '--output', type = str, 
        default = "./pruned_model.pt",
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

    # ダミー入力
    data_list = torch.load(data_path, weights_only = False)
    loader = DataLoader(data_list, batch_size = 1, shuffle =  False)
    data = next(iter(loader))
    data = data.to(device)
    print(f"ダミー入力を{data_path}から読み込みました。")

    example_inputs=(data.x, data.edge_index, data.edge_weight.requires_grad_())

    total_energy = model(*example_inputs)
    print("pruning前: ")
    print(f"total_energy: {total_energy}")

    # プルーニング
    imp = tp.importance.GroupMagnitudeImportance(p=2)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 1:
            ignored_layers.append(m)

    pruner = tp.pruner.MagnitudePruner(
        model, 
        example_inputs=example_inputs, 
        importance=imp, 
        pruning_ratio=0.5, 
        ignored_layers=ignored_layers, 
        round_to=8
    )

    base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    tp.utils.print_tool.before_pruning(model) # or print(model)
    pruner.step()
    tp.utils.print_tool.after_pruning(model) # or print(model), this util will show the difference before and after pruning
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print(f"MACs: {base_macs/1e9} G -> {macs/1e9} G, #Params: {base_nparams/1e6} M -> {nparams/1e6} M")

    total_energy = model(*example_inputs)
    print("pruning後: ")
    print(f"total_energy: {total_energy}")

    torch.save(model, output_path)

if __name__ == "__main__":
    main()