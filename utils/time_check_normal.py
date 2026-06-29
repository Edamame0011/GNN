from model.schnet_model_export import SchNetModel
import torch
import argparse
from train.train_schnet import load_config
import time

def main():
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = "入力モデルのパス"
    )
    parser.add_argument(
        '--config', type = str, 
        default = "./configs/config.json", 
        help = "configファイルのパス"
    )
    args = parser.parse_args()

    input_path = args.input
    config_path = args.config

    # configファイルの読み込み
    config = load_config(config_path)
    num_interactions    = config["num_interactions"]
    cutoff              = config["cutoff"]
    num_gaussians       = config["num_gaussians"]
    hidden_dim          = config["hidden_dim"]
    num_filters         = config["num_filters"]

    device = torch.device("cuda")

    # モデルの読み込み
    model = SchNetModel(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    model.to(device)
    model.eval()

    num_nodes = 50
    num_edges = 1000

    x = torch.randint(0, 100, (num_nodes, ), dtype=torch.long, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long, device=device)
    edge_weight = torch.randn((3, num_edges), dtype=torch.float, device=device).requires_grad_(True)

    for _ in range(10):
        _ = model(x, edge_index, edge_weight)
    
    print("計測開始")

    start_time = time.time()

    for _ in range(1000):
        _ = model(x, edge_index, edge_weight)
    
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    average_time = (total_time / 1000) * 1000

    print(f"トータル時間 (1000回): {total_time:.4f} 秒")
    print(f"1回あたりの平均時間: {average_time:.4f} ミリ秒 (ms)")

if __name__ == "__main__":
    main()