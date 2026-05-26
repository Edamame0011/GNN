import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import argparse
from model.schnet_model_export import SchNetModel
from train.train_schnet import load_config

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

    device = torch.device("cuda")

    # モデルの読み込み
    model = SchNetModel(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    state_dict = torch.load(input_path, weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # プルーニング
    amount = 0.3
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)

            if module.bias is not None:
                prune.l1_unstructured(module, name="bias", amount=amount)
            
            prune.remove(module, "weight")
            if module.bias is not None:
                prune.remove(module, "bias")
    

        #スクリプト化
    script_module = torch.jit.script(model)

    #モデルの保存
    script_module.save(output_path)

if __name__ == "__main__":
    main()