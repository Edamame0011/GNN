データの前処理  
```python -m utils.preprocess path/to/xyz_file.xyz```


学習  
```python -m train.train_schnet --config path/to/config.json```  


デプロイ  
```python -m utils.deploy path/to/model.pth -o path/to/output.pt```
