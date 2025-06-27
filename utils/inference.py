import torch
from torch_geometric.loader import DataLoader
from train.train_schnet import load_config

config_path = "configs/config.json"
model_path = "model_schnet_full.pth"

def main():
    #configファイルの読み込み
    config = load_config(config_path)

    data_path           = config["data_path"]

    #データの読み込み
    data_list = torch.load(data_path, weights_only = False)
    dataloader = DataLoader(data_list, shuffle = False, num_workers = 4)
    print("データを読み込みました。")

    #モデルの読み込み
    model = torch.load(model_path, weights_only = False)
    print("モデルを読み込みました。")
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #推論
    results_energy=[]
    results_forces=[]
    for batch in dataloader:
        batch = batch.to(device)
        energies,forces = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        results_energy.append(energies.detach().to('cpu').item())
        arr_force = forces.detach().to('cpu').numpy()
        results_forces.append(arr_force)

    with open("output.txt", 'w') as f:
        for idx in range(len(results_energy)):
            f.write(f"structure[{idx}]\n")
            f.write(f"potential_energy: {results_energy[idx]}\n")

            # forces は numpy の 2D 配列になっているので、原子ごとに出力
            f.write("forces:\n")
            force_arr = results_forces[idx]  # 形状: (n_atoms, 3)
            for atom_idx, vec in enumerate(force_arr):
                fx, fy, fz = vec.tolist()
                f.write(f"  atom[{atom_idx}]: [{fx:.15f}, {fy:.15f}, {fz:.15f}]\n")
            f.write("\n")
            
if __name__ == '__main__':
    main()