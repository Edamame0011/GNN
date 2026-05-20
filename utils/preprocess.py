import torch
import numpy as np
from matscipy.neighbours import neighbour_list
import argparse
from torch_geometric.data import Data
from torch import save
from ase.io import iread
from tqdm import tqdm

class CustomData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ["forces", "pos", "edge_weight"]:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

def AtomsToCustomData(atoms, cutoff):
    atoms.set_pbc([True, True, True]) 

    x = torch.tensor(atoms.numbers, dtype = torch.int64)                    # (N, )
    y = torch.tensor(atoms.get_potential_energy(), dtype = torch.float32)   # (1, )
    forces = torch.tensor(atoms.get_forces(), dtype = torch.float32).t()    # (3, N)
    pos = torch.tensor(atoms.get_positions(), dtype = torch.float32).t()    # (3, N)

    #近接ペアと距離情報を取得
    #i: ソース原子のインデックス (N, )
    #j: ターゲット原子のインデックス (N, )
    #D: 原子間距離ベクトル (N, 3)
    i, j, D = neighbour_list('ijD', atoms, cutoff = cutoff)

    # iについてソート（CSR対応のため）
    sort_idx = np.lexsort((j, i)) 
    
    i = i[sort_idx]
    j = j[sort_idx]
    D = D[sort_idx]

    i = np.array(i, dtype = np.int64)
    j = np.array(j, dtype = np.int64)
    D = np.array(D, dtype = np.float32)

    edge_weight = torch.tensor(D, dtype = torch.float32).t()            # (3, num_edges)
    edge_index = torch.tensor(np.stack([i, j]), dtype = torch.int64)    # (2, num_edges)

    data = CustomData(
        x = x, 
        y = y, 
        forces = forces, 
        pos = pos, 
        edge_index = edge_index, 
        edge_weight = edge_weight
    )

    return data

#原子リストをデータリストに変換
def ConvertAtomListToDataList(atoms_list, cutoff):
    data_list = []
    for atoms in tqdm(atoms_list):
        data = AtomsToCustomData(atoms, cutoff)
        data_list.append(data)

    return data_list

def main():
    #コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type = str, 
        help = ".xyzファイルのパス"
    )
    parser.add_argument(
        '--cutoff', type = float, 
        default = 5.0, 
        help = "カットオフ半径"
    )
    args = parser.parse_args()

    path = args.input
    cutoff = args.cutoff

    atoms = iread(path, format = 'extxyz')
    atoms_list = []

    for atom in atoms:
        atoms_list.append(atom)

    data_list = ConvertAtomListToDataList(atoms_list, cutoff)

    save(data_list, 'data/data.pt')

if __name__ == '__main__':
    main()