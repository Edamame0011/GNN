import torch
from torch_geometric.loader import DataLoader
from model.SchNetCalculator import convert
from ase.io import read
import torch
import argparse
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader
from utils.preprocess import CustomData
from model.schnet_model import SchNetModel

def main():
    #コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = "モデルのパス"
    )
    parser.add_argument(
        '-o', '--output', type = str, 
        default = "./output.txt",
        help = "変換後モデルの保存先"
    )
    parser.add_argument(
        '--config', type = str, 
        default = "./configs/config.json", 
        help = "configファイルのパス"
    )
    parser.add_argument(
        "--structure", type=str, 
        default="./data/sample_NS2.xyz", 
        help="構造のパス"
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
    structure_path = args.structure

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #モデルの読み込み
    model = SchNetModel(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    state_dict = torch.load(input_path, weights_only=False)["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    #データの読み込み
    atoms = read(structure_path, format = 'extxyz')
    data = convert(atoms, cutoff = 5.0)
    print("データを読み込みました。")

    #推論
    data = data.to(device)
    energies,forces = model(data.x, data.edge_index, data.edge_weight)
    result_energy = energies.detach().to('cpu').item()
    result_force = forces.detach().to('cpu').t().numpy()

    with open(output_path, 'w') as f:
        f.write(f"potential_energy: {result_energy}\n")
        # forces は numpy の 2D 配列になっているので、原子ごとに出力
        f.write("forces:\n")  # 形状: (3, n_atoms)
        for atom_idx, vec in enumerate(result_force):
            fx, fy, fz = vec.tolist()
            f.write(f"  atom[{atom_idx}]: [{fx:.15f}, {fy:.15f}, {fz:.15f}]\n")
        f.write("\n")
            
if __name__ == '__main__':
    main()