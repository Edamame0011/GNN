import torch
import numpy as np
import random
import json
from model.schnet_model import SchNetModel
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import argparse

#シードを設定する関数
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#訓練ループ
def train(model, criterion, energy_weight, force_weight, dataloader, optimizer, device):
    model.train()
    loss_total = 0
    loss_e_total = 0
    loss_f_total = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        energies, forces = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

        loss_e = criterion(energies, batch.y)
        loss_f = criterion(forces, batch.forces)

        l = loss_e * energy_weight + loss_f * force_weight
        l.backward()
        optimizer.step()

        loss_total += l.item()
        loss_e_total += loss_e.item()
        loss_f_total += loss_f.item()
    
    loss_total = loss_total / len(dataloader)
    loss_e_total = loss_e_total / len(dataloader)
    loss_f_total = loss_f_total / len(dataloader)
    
    return loss_total, loss_e_total, loss_f_total

#評価ループ
def evaluate(model, criterion, energy_weight, force_weight, dataloader, device):
    model.eval()
    loss_total = 0
    loss_e_total = 0
    loss_f_total = 0

    for batch in dataloader:
        batch = batch.to(device)
        energies, forces = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

        loss_e = criterion(energies, batch.y)
        loss_f = criterion(forces, batch.forces)

        l = loss_e * energy_weight + loss_f * force_weight

        loss_total += l.item()
        loss_e_total += loss_e.item()
        loss_f_total += loss_f.item()
    
    loss_total = loss_total / len(dataloader)
    loss_e_total = loss_e_total / len(dataloader)
    loss_f_total = loss_f_total / len(dataloader)

    return loss_total, loss_e_total, loss_f_total

#データセットの読み込み
def make_dataloaders(path, batch_size):
    data_list = torch.load(path, weights_only = False)
    train_data, test_data = train_test_split(data_list, test_size = 0.2)

    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = 4)

    return train_dataloader, test_dataloader

#configファイルの読み込み
def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)

        return config

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
    energy_weight       = config["energy_weight"]
    force_weight        = config["force_weight"]
    cutoff              = config["cutoff"]
    num_gaussians       = config["num_gaussians"]
    hidden_dim          = config["hidden_dim"]
    num_filters         = config["num_filters"]
    data_path           = config["data_path"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #シードの設定
    set_seed(42)
    
    #データセットの読み込み
    train_dataloader, test_dataloader = make_dataloaders(data_path, batch_size)

    #モデルの作成
    model = SchNetModel(hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
                        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff)
    model = model.to(device)

    force_weight = torch.tensor(force_weight).to(device)
    energy_weight = torch.tensor(energy_weight).to(device)

    #モデルの学習
    writer = SummaryWriter()
    optimizer = AdamW(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.5)
    criterion = MSELoss()

    for epoch in range(epochs):
        #学習
        loss_total, loss_e_total, loss_f_total = train(model = model, 
                                                       criterion = criterion, 
                                                       energy_weight = energy_weight, 
                                                       force_weight = force_weight, 
                                                       dataloader = train_dataloader, 
                                                       optimizer = optimizer, 
                                                       device = device)
        scheduler.step()
        print('epoch: train', epoch, 'loss_total', loss_total, 'loss_e', loss_e_total, 'loss_f', loss_f_total)
        writer.add_scalar('loss_total', loss_total, epoch)
        writer.add_scalar('loss_e', loss_e_total, epoch)
        writer.add_scalar('loss_f', loss_f_total, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        #評価
        loss_total, loss_e_total, loss_f_total = evaluate(model = model, 
                                                          criterion = criterion, 
                                                          energy_weight = energy_weight, 
                                                          force_weight = force_weight, 
                                                          dataloader = test_dataloader, 
                                                          device = device)

        print('epoch: test', epoch, 'loss_total', loss_total, 'loss_e', loss_e_total, 'loss_f', loss_f_total)
        writer.add_scalar('loss_total_test', loss_total, epoch)
        writer.add_scalar('loss_e_test', loss_e_total, epoch)
        writer.add_scalar('loss_f_test', loss_f_total, epoch)

    writer.close()

    #モデルの保存
    torch.save({'model_state_dict': model.state_dict(),
           'setups': model.setups}, 'model_schnet.pth')
    torch.save(model, 'model_schnet_full.pth')

    #プロット 
    results_energy=[]
    results_forces=[]
    ref_energy=[]
    ref_forces=[]
    for batch in test_dataloader:
        batch = batch.to(device)
        energies,forces = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        results_energy.extend(energies.detach().to('cpu').numpy().flatten())
        results_forces.extend(forces.detach().to('cpu').numpy().flatten())
        ref_energy.extend(batch.y.detach().to('cpu').numpy().flatten())
        ref_forces.extend(batch.forces.detach().to('cpu').numpy().flatten())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ref_energy,results_energy)
    plt.xlabel('reference energy')
    plt.ylabel('predicted energy')
    plt.title('SchNetModel energy')
    plt.savefig('energy_schnet_torch_full_stepLR.png')
    plt.close()

    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ref_forces,results_forces)
    plt.xlabel('reference forces')
    plt.ylabel('predicted forces')
    plt.title('SchNetModel forces')
    plt.savefig('forces_schnet_torch_full_stepLR.png')
    plt.close()

if __name__ == '__main__':
    main()