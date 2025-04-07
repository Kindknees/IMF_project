# 如何複製環境並執行：

1. 先將my_env.tar.gz下載到這個資料夾，接著進行解壓縮就會複製整個環境了。my_env.tar.gz包含了整個環境的所有套件內容，因此應該不是連網安裝，而是直接包在裡面的：
```
tar -xzvf my_env.tar.gz -C conda環境
# 範例：tar -xzvf my_env.tar.gz -C /opt/homebrew/Caskroom/miniconda/envs
```

2. 啟動環境：
```
conda activate 環境名稱
```
