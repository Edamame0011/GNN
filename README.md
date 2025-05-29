データの前処理  
```python -m utils.preprocess <xyzファイルへのパス>```


学習  
```python -m train.train_schnet --config <configファイルへのパス>```  


デプロイ  
```python -m utils.deploy <モデルへのパス> -o <出力モデル名>```
