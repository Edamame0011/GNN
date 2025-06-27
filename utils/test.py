import torch
from torch_geometric.loader import DataLoader

data_path = "data/data.pt"

def main():
    #データの読み込み
    data_list = torch.load(data_path, weights_only = False)
    dataloader = DataLoader(data_list, shuffle = False, num_workers = 4)
    print("データを読み込みました。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #保存
    for batch in dataloader:
        batch = batch.to(device)
        torch.save(batch.x, "x.pt")
        torch.save(batch.edge_index, "edge_index.pt")
        torch.save(batch.edge_weight, "edge_weight.pt")
        break
        

if __name__ == "__main__":
    main()