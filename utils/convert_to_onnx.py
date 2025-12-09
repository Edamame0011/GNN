import torch
import argparse
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader

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
        help = "configファイルのパス（ダミー入力の作成に使う）"
    )
    args = parser.parse_args()

    #configファイルの読み込み
    config_path =  "configs/config.json"
    config = load_config(config_path)
    data_path = config["data_path"]

    input_path = args.input
    output_path = args.output

    device = torch.device("cpu")

    #モデルの読み込み
    model = torch.load(input_path, weights_only = False)
    model.to(device)
    model.eval()

    #ダミー入力
    data_list = torch.load(data_path, weights_only = False)
    loader = DataLoader(data_list, batch_size = 1, shuffle =  False)
    data = next(iter(loader))
    example_inputs = (data.x, data.edge_index, data.edge_weight, data.batch)

    #エクスポート
    torch.onnx.export(
        model, 
        example_inputs, 
        output_path, 
        export_params=True, 
        opset_version=13, 
        do_constant_folding=True, 
        input_names=["atomic_numbers", "edge_index", "edge_weight"], 
        output_names=["energy", "force"]
    )