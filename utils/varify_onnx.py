import torch
import numpy as np
import onnxruntime as ort
import argparse
from train.train_schnet import load_config
from torch_geometric.loader import DataLoader
from utils.preprocess import CustomData
# モデルのインポート（パスは環境に合わせてください）
from model.schnet_model import SchNetModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default="schnet_model.onnx", help="エクスポートしたONNXモデルのパス")
    parser.add_argument('--pt', type=str, required=True, help="元のPyTorchモデルの重み(.pt)のパス")
    parser.add_argument('--config', type=str, default="./configs/config.json", help="configファイルのパス")
    args = parser.parse_args()

    # 1. Configとデータの読み込み
    config = load_config(args.config)
    device = torch.device("cpu") # 誤差比較のためCPUで実行します

    data_list = torch.load(config["data_path"], weights_only=False)
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    data = next(iter(loader))
    data.to(device)

    print("=== PyTorchモデルでの推論 (Autograd) ===")
    
    # 2. PyTorchモデルの準備
    pt_model = SchNetModel(
        hidden_dim=config["hidden_dim"], num_gaussians=config["num_gaussians"], 
        num_filters=config["num_filters"], num_interactions=config["num_interactions"], cutoff=config["cutoff"]
    )
    state_dict = torch.load(args.pt, map_location=device, weights_only=False)["model_state_dict"]
    pt_model.load_state_dict(state_dict)
    pt_model.eval()

    # Autogradで力を計算するために勾配追跡を有効化
    data.edge_weight.requires_grad_(True)

    # エネルギーの計算
    outputs = pt_model(data.x, data.edge_index, data.edge_weight)

    pt_energy_np = outputs[0].detach().numpy()
    pt_forces_np = outputs[1].detach().numpy()
    
    print(f"PyTorch Energy: {pt_energy_np}")
    print(f"PyTorch Forces shape: {pt_forces_np.shape}")


    print("\n=== ONNX Runtimeでの推論 ===")
    
    # 3. ONNX Runtime セッションの初期化
    # プロバイダとしてCPUを指定（CUDA環境の動作確認をしたい場合は 'CUDAExecutionProvider' を追加）
    ort_session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
    
    # ONNXの入力名を取得 (Dynamoエクスポートでは自動生成名になるため動的に取得)
    input_names = [inp.name for inp in ort_session.get_inputs()]
    
    # 入力をNumPy配列に変換
    # ONNXはPyTorchのTensorを直接受け取れないため、numpy化します
    x_np = data.x.detach().numpy()
    edge_index_np = data.edge_index.detach().numpy()
    edge_weight_np = data.edge_weight.detach().numpy()

    # 入力辞書の作成
    ort_inputs = {
        input_names[0]: x_np,
        input_names[1]: edge_index_np,
        input_names[2]: edge_weight_np
    }

    # 推論実行
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Dynamoは出力の順番を内部ロジックで決めるため、取り出して確認
    ort_energy_np = ort_outs[0]
    ort_forces_np = ort_outs[1]

    print(f"ONNX Energy: {ort_energy_np}")
    print(f"ONNX Forces shape: {ort_forces_np.shape}")

    print("\n=== 誤差検証 ===")
    
    # 4. 誤差の比較
    # Float32の演算精度に起因する微小な誤差(1e-5前後)は許容します
    energy_diff = np.abs(pt_energy_np - ort_energy_np).max()
    forces_diff = np.abs(pt_forces_np - ort_forces_np).max()

    print(f"エネルギーの最大絶対誤差: {energy_diff:.2e}")
    print(f"力の最大絶対誤差       : {forces_diff:.2e}")

    # numpyのテスト関数で許容範囲内か厳密にチェック (1e-4程度を許容)
    try:
        np.testing.assert_allclose(pt_energy_np, ort_energy_np, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(pt_forces_np, ort_forces_np, rtol=1e-4, atol=1e-4)
        print("\n✅ テスト成功: PyTorchとONNXの出力は数学的に一致しています！")
    except AssertionError as e:
        print("\n❌ テスト失敗: 出力に許容範囲を超える誤差があります。")
        print(e)

if __name__ == "__main__":
    main()