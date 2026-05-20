import torch
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader
from model.schnet_model import SchNetModel
from utils.preprocess import CustomData
import argparse
import os
from torch.export import draft_export

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
    batch_size          = config["batch_size"]
    lr                  = config["lr"]
    epochs              = config["epochs"]
    num_interactions    = config["num_interactions"]
    energy_weight       = config["energy_weight"]
    force_weight        = config["force_weight"]
    cutoff              = config["cutoff"]
    num_gaussians       = config["num_gaussians"]
    hidden_dim          = config["hidden_dim"]
    num_filters         = config["num_filters"]
    data_path           = config["data_path"]
    output_name         = config["output_name"]

    device = torch.device("cuda")

    # モデルの読み込み
    model = SchNetModel(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    state_dict = torch.load(input_path, weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    # ダミー入力
    data_list = torch.load(data_path, weights_only = False)
    loader = DataLoader(data_list, batch_size = 1, shuffle =  False)
    data = next(iter(loader))
    data.to(device)
    example_inputs = (data.x, data.edge_index, data.edge_weight)
    print(f"ダミー入力を{data_path}から読み込みました。")

    # 要素数が可変な要素の設定
    num_nodes = torch.export.Dim("num_nodes", min=1)
    num_edges = torch.export.Dim("num_edges", min=1)
    
    dynamic_shapes = (
        {0: num_nodes}, # data.x (num_nodes, )
        {1: num_edges}, # data.edge_index (2, num_edges)
        {1: num_edges}, # data.edge_weight (3, num_edges)
    )

    # エクスポート
    exported_program = torch.export.draft_export(
        model, 
        args=example_inputs, 
        dynamic_shapes=dynamic_shapes
    )
    torch._inductor.aoti_compile_and_package(
        exported_program, 
        package_path=output_path, 
        inductor_configs={"max_autotune": True}
    )
    print(f"モデルが正常に {output_path} へエクスポートされました。")

if __name__ == "__main__":
    main()