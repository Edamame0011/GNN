import torch
import numpy as np
import random
import json
from model.gnnff_energy_model import GNNFF_train
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import argparse
from utils.preprocess import CustomData
from train.train_gnnff_train import load_config, make_dataloaders

def main():
    #コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        default = "./configs/config.json", 
        help = "configファイルへのパス"
    )
    args = parser.parse_args()

    #configファイルの読み込み
    config_path =  args.config
    config = load_config(config_path)

    batch_size          = config["batch_size"]
    lr                  = config["lr"]
    epochs              = config["epochs"]
    num_interactions    = config["num_interactions"]
    cutoff              = config["cutoff"]
    num_gaussians       = config["num_gaussians"]
    hidden_dim          = config["hidden_dim"]
    num_filters         = config["num_filters"]
    data_path           = config["data_path"]
    output_name         = config["output_name"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #データセットの読み込み
    train_dataloader, test_dataloader = make_dataloaders(data_path, batch_size)

    #モデルの作成
    model = torch.load("model_gnnff_full.pth", weights_only=False)

    #プロット 
    results_energy=[]
    results_forces=[]
    results_forces_pred=[]
    ref_energy=[]
    ref_forces=[]
    for batch in test_dataloader:
        batch = batch.to(device)
        forces_pred, energies,forces = model(batch.x, batch.edge_index, batch.edge_weight, True, batch.batch)
        results_energy.extend(energies.detach().to('cpu').numpy().flatten())
        results_forces.extend(forces.detach().to('cpu').numpy().flatten())
        results_forces_pred.extend(forces_pred.detach().to('cpu').numpy().flatten())
        ref_energy.extend(batch.y.detach().to('cpu').numpy().flatten())
        ref_forces.extend(batch.forces.detach().to('cpu').numpy().flatten())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ref_energy,results_energy)
    plt.xlabel('reference energy')
    plt.ylabel('predicted energy')
    plt.title('SchNetModel energy')
    plt.savefig(f'energy_{output_name}_torch_full_stepLR.png')
    plt.close()

    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ref_forces,results_forces)
    plt.xlabel('reference forces')
    plt.ylabel('predicted forces')
    plt.title('SchNetModel forces')
    plt.savefig(f'forces_{output_name}_torch_full_stepLR.png')
    plt.close()

    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ref_forces,results_forces_pred)
    plt.xlabel('reference forces')
    plt.ylabel('predicted forces')
    plt.title('SchNetModel forces')
    plt.savefig(f'forces_pred_{output_name}_torch_full_stepLR.png')
    plt.close()

if __name__ == "__main__":
    main()