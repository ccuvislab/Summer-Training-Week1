### nvidia-smi 截圖
![nvidia-smi](nvidia-smi-screenshot.png)

### tmux 畫面
![tmux](tmux_screenshot.png)

### wandb 畫面
![wandb](wandb_screenshot.png)

### 用 salloc 訓練程式
![salloc](salloc_screenshot.png)
在 salloc 的節點中執行 python train.py 測試訓練程式

### 用 sbatch 派送任務，並用 tail 指令監控訓練過程
![sbatch](sbatch_screenshot.png)
先用 sbatch 發送任務，在輸入 tail fashion_{job_id}.out 監控訓練過程