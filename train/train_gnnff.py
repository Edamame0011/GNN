import torch
import numpy as np
import random
import json
from model.gnnff_model import GNNFF
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.optim import AdamW
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import argparse
from utils.preprocess import CustomData

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
def train(model, criterion, dataloader, optimizer, device):
    model.train()
    loss_total = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        forces = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

        l = criterion(forces, batch.forces)

        l.backward()
        optimizer.step()

        loss_total += l.item()
    
    loss_total = loss_total / len(dataloader)
    
    return loss_total

#評価ループ
def evaluate(model, criterion, dataloader, device):
    model.eval()
    loss_total = 0

    for batch in dataloader:
        batch = batch.to(device)
        forces = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)

        l = criterion(forces, batch.forces)

        loss_total += l.item()
    
    loss_total = loss_total / len(dataloader)

    return loss_total

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
    cutoff              = config["cutoff"]
    num_gaussians       = config["num_gaussians"]
    hidden_dim          = config["hidden_dim"]
    num_filters         = config["num_filters"]
    data_path           = config["data_path"]
    output_name         = config["output_name"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #シードの設定
    set_seed(42)
    
    #データセットの読み込み
    train_dataloader, test_dataloader = make_dataloaders(data_path, batch_size)

    #モデルの作成
    model = GNNFF(
        hidden_dim = hidden_dim, num_gaussians = num_gaussians, 
        num_filters = num_filters, num_interactions = num_interactions, cutoff = cutoff
    )
    model = model.to(device)

    #モデルの学習
    writer = SummaryWriter()
    optimizer = AdamW(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.5)
    criterion = MSELoss()

    for epoch in range(epochs):
        #学習
        loss_total = train(
            model = model, 
            criterion = criterion, 
            dataloader = train_dataloader, 
            optimizer = optimizer, 
            device = device
        )
        scheduler.step()
        print('epoch: train', epoch, ', loss_total: ', loss_total)
        writer.add_scalar('loss_total', loss_total, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        #評価
        loss_total = evaluate(
            model = model, 
            criterion = criterion, 
            dataloader = test_dataloader, 
            device = device
        )

        print('epoch: test', epoch, ', loss_total: ', loss_total,)
        writer.add_scalar('loss_total_test', loss_total, epoch)

    writer.close()

    save_path = output_name + ".pth"
    save_path_full = output_name + "_full.pth"

    #モデルの保存
    torch.save({'model_state_dict': model.state_dict(),
           'setups': model.setups}, save_path)
    torch.save(model, save_path_full)

    #プロット 
    results_forces=[]
    ref_forces=[]
    for batch in test_dataloader:
        batch = batch.to(device)
        forces = model(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
        results_forces.extend(forces.detach().to('cpu').numpy().flatten())
        ref_forces.extend(batch.forces.detach().to('cpu').numpy().flatten())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,5))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.scatter(ref_forces,results_forces)
    plt.xlabel('reference forces')
    plt.ylabel('predicted forces')
    plt.title('SchNetModel forces')
    plt.savefig(f'forces_{output_name}_torch_full_stepLR.png')
    plt.close()

if __name__ == '__main__':
    main()