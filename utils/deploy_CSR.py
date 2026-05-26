import torch
import argparse
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader
from utils.preprocess import CustomData
from model.schnet_model_CSR_export import SchNetModel

def main():
    #コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = "変換前モデルのパス"
    )
    parser.add_argument(
        '-o', '--output', type = str, 
        default = "./deployed_model.pt",
        help = "変換後モデルの保存先"
    )
    parser.add_argument(
        '--config', type = str, 
        default = "./configs/config.json", 
        help = "configファイルのパス"
    )
    args = parser.parse_args()

    #configファイルの読み込み
    config_path =  "configs/config.json"
    config = load_config(config_path)
    num_interactions    = config["num_interactions"]
    cutoff              = config["cutoff"]
    num_gaussians       = config["num_gaussians"]
    hidden_dim          = config["hidden_dim"]
    num_filters         = config["num_filters"]

    input_path = args.input
    output_path = args.output

    device = torch.device("cpu")

    #モデルの読み込み
    model = SchNetModel(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    state_dict = torch.load(input_path, weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    #スクリプト化
    script_module = torch.jit.script(model)

    #モデルの保存
    script_module.save(output_path)

if __name__ == "__main__":
    main()