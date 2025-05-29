import torch
import numpy as np
from matscipy.neighbours import neighbour_list

#原子の近傍情報を計算し、原子の近接インデックスとエッジ特徴量（距離）に返還
def RadiusInteractionGraph(atoms, cutoff):
    """
    Args:
        atoms: ASEのAtomsオブジェクト
        cutoff: カットオフ距離
    
    Returns:
        edge_index: 近接ペアのインデックスを表すテンソル、サイズは(2, num_edges)
        edge_weight: 原子間距離ベクトルを表すテンソル、サイズは(num_edges, 3)
    """

    #近接ペアと距離情報を取得
    #i: ソース原子のインデックス (num_edges, )
    #j: ターゲット原子のインデックス (num_edges, )
    #原子i[k]とj[k]が接続している
    #D: 原子間距離ベクトル (num_edges, 3)
    i, j, D = neighbour_list('ijD', atoms, cutoff = cutoff)

    i = np.array(i, dtype = np.int64)
    j = np.array(j, dtype = np.int64)
    D = np.array(D, dtype = np.float32)

    #原子間距離ベクトル (3, num_edges)
    edge_weight = torch.tensor(D, dtype = torch.float32)
    #近接ペアのインデックス (2, num_edges)
    #np.stack()は複数配列を新しい軸でまとめる関数（np.concatenate()は単につなげる関数）
    edge_index = torch.tensor(np.stack([i, j]), dtype = torch.int64)

    return edge_index, edge_weight

from torch_geometric.data import Data

#ASEのAtomsオブジェクトをPyGのDataオブジェクト（グラフ）に変換
def AtomsToPyGData(atoms, cutoff):
    """
    Args:
        atoms: ASEのAtomsオブジェクト
        cutoff: カットオフ距離
    
    Returns:
        data: PyGのDataオブジェクト
    """
    #それぞれの原子の原子番号 (N, 1)
    x = torch.tensor(atoms.numbers, dtype = torch.int64)
    #系のエネルギー (1, )
    y = torch.tensor(atoms.get_potential_energy(), dtype = torch.float32)
    #それぞれの原子の力 (N, 3)
    forces = torch.tensor(atoms.get_forces(), dtype = torch.float32)
    #それぞれの原子の座標 (N, 3)
    pos = torch.tensor(atoms.get_positions(), dtype = torch.float32)

    #グラフの接続情報とエッジ特徴量（距離）
    edge_index, edge_weight = RadiusInteractionGraph(atoms, cutoff)

    data = Data(
        x = x,                          #ノード特徴量 (num_nodes, num_features)
        y = y,                          #ラベル (1, )
        forces = forces,                
        edge_index = edge_index,        #エッジの接続情報 (2, num_edges)
        edge_weight = edge_weight,      
        pos = pos                       #ノードの位置 (num_nodes, 3)
        #edge_attr = edge_attr          #エッジ特徴量 (num_edges, edge_feat_dim)
        )
    
    return data

#原子リストをデータリストに変換
def ConvertAtomListToDataList(atoms_list, cutoff):
    data_list = []
    for atoms in atoms_list:
        data = AtomsToPyGData(atoms, cutoff)
        data_list.append(data)

    return data_list

from torch import save
from ase.io import iread

def main():
    path = "data/sample.xyz"
    cutoff = 5.0

    atoms = iread(path, format = 'extxyz')
    atoms_list = []

    for atom in atoms:
        atoms_list.append(atom)

    data_list = ConvertAtomListToDataList(atoms_list, cutoff)

    save(data_list, 'data/data.pt')

if __name__ == '__main__':
    main()