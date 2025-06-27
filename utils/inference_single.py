import torch
from torch_geometric.loader import DataLoader
from model.SchNetCalculator import convert
from ase.io import read

config_path = "configs/config.json"
model_path = "model_schnet_full.pth"

def main():
    structure_path = "data/initial_structure.xyz"

    #データの読み込み
    atoms = read(structure_path, format = 'extxyz')
    data = convert(atoms, cutoff = 5.0)
    print("データを読み込みました。")

    #モデルの読み込み
    model = torch.load(model_path, weights_only = False)
    print("モデルを読み込みました。")
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #推論
    data = data.to(device)
    energies,forces = model(data.x, data.edge_index, data.edge_weight)
    result_energy = energies.detach().to('cpu').item()
    result_force = forces.detach().to('cpu').numpy()

    with open("output.txt", 'w') as f:
        f.write(f"potential_energy: {result_energy}\n")
        # forces は numpy の 2D 配列になっているので、原子ごとに出力
        f.write("forces:\n")  # 形状: (n_atoms, 3)
        for atom_idx, vec in enumerate(result_force):
            fx, fy, fz = vec.tolist()
            f.write(f"  atom[{atom_idx}]: [{fx:.15f}, {fy:.15f}, {fz:.15f}]\n")
        f.write("\n")
            
if __name__ == '__main__':
    main()