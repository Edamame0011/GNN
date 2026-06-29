import torch
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader
import torch.nn as nn
from model.gnnff_model import GNNFF
from utils.preprocess import CustomData
import argparse
from torch.func import functional_call, grad
from torch.fx.experimental.proxy_tensor import make_fx

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
    model = GNNFF(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    state_dict = torch.load(input_path, weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

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
    )

    # エクスポート
    exported_program = torch.export.export(
        model, 
        args=(data.x, data.edge_index, data.edge_weight), 
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