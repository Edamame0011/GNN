import torch
from utils.preprocess import RadiusInteractionGraph
from ase.io import read
import pickle

atoms = read("data/initial_structure.xyz", format = "extxyz")

edge_index, edge_weight = RadiusInteractionGraph(atoms, 5.0)


torch.save(edge_index, "edge_index.pt")
torch.save(edge_weight, "edge_weight.pt")