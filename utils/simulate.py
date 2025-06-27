from model.SchNetCalculator import SchNetCalculator
from ase.io import read
from ase import units
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import torch

data_path = "data/initial_structure.xyz"
model_path = "model_schnet_full.pth"
cutoff = 5.0
timestep = 0.005 * units.fs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#アウトプットの設定
interval_step = 100
log_path = "log/md.log"

#変数の初期化
atoms = read(data_path)
model = torch.load(model_path, weights_only = False)
calculator = SchNetCalculator(model, cutoff, device)

#calculatorをセット
atoms.set_calculator(calculator)

#初期速度の設定
MaxwellBoltzmannDistribution(atoms, temperature_K = 300)

#シミュレーションの実体化
dyn = VelocityVerlet(atoms, timestep, logfile = log_path, loginterval = interval_step)

#シミュレーションの実行（引数：ステップ数）
dyn.run(1e+5)