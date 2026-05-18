from ase.calculators.calculator import Calculator, all_changes
from utils.preprocess import CustomData
from matscipy.neighbours import neighbour_list
import numpy as np
import torch
from torch_geometric.data import Data

def convert(atoms, cutoff):
    atoms.set_pbc([True, True, True]) 

    x = torch.tensor(atoms.numbers, dtype = torch.int64)

    i, j, D = neighbour_list('ijD', atoms, cutoff = cutoff)

    i = np.array(i, dtype = np.int64)
    j = np.array(j, dtype = np.int64)
    D = np.array(D, dtype = np.float32)

    edge_weight = torch.tensor(D, dtype = torch.float32).t()            # (3, num_edges)
    edge_index = torch.tensor(np.stack([i, j]), dtype = torch.int64)    # (2, num_edges)

    data = CustomData(
        x = x, 
        edge_index = edge_index, 
        edge_weight = edge_weight
    )

    return data

class SchNetCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    #コンストラクタ
    def __init__(self, model, cutoff, device = 'cpu'):
        Calculator.__init__(self)
        self.model = model.to(device)
        self.cutoff = cutoff
        self.device = device
    
    def calculate(self, atoms = None, properties = ['energy'], system_changes = all_changes):
        """
        原子の、指定された物理量を計算する。
        
        Args:
            atoms: 原子
            properties: 計算する物理量
            system_changes: 
        """

        #指定した物理量の計算が必要か？
        if self.calculation_required(atoms, properties):
            self.results = {}

            #原子をグラフに変換
            #AtomsToPyGData()を使うと、中でatoms.get_potential()が呼ばれるため使わない
            data = convert(atoms, self.cutoff)
            data = data.to(self.device)
            #力の計算のためedge_weightの勾配計算をする
            data.edge_weight.requires_grad = True

            #エネルギーと力をモデルで推論
            energy, force = self.model(data.x, data.edge_index, data.edge_weight)

            #結果を表す辞書型変数resultsに格納
            self.results['energy'] = energy.to('cpu').item()
            self.results['forces'] = force.to('cpu').detach().numpy()