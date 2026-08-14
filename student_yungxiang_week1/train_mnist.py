# 1️⃣ Import necessary libraries / 載入必要的套件
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb

# 2️⃣ Initialize wandb with project name and config / 初始化 wandb，設定專案名稱與超參數設定
wandb.init(
    project="mnist_mlp_demo",
    config={
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 64,
        "hidden_dims": [256, 128, 64],  # 代表有 3 層隱藏層
    }
)
config = wandb.config

# 3️⃣ Set device (GPU if available) / 設定裝置（優先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 4️⃣ Load dataset with transform / 載入資料集並套用轉換（transform）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的平均值與標準差標準化
])

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

# 5️⃣ Create DataLoader for batch processing / 建立 DataLoader 進行批次處理
loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True
)

# 6️⃣ Define MLP model (支援可自訂層數與維度)
class MLP(nn.Module):
    def __init__(self, input_dim=28*28, output_dim=10, hidden_dims=[128, 128]):
        super().__init__()
        layers = [nn.Flatten()]  # 自動將 28x28 展平

        prev_dim = input_dim
        # 根據 hidden_dims 的長度動態疊加隱藏層
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # 輸出層
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 初始化範例：想開 3 層隱藏層就傳入長度為 3 的列表
model = MLP(hidden_dims=wandb.config.hidden_dims).to(device)

# 7️⃣ Define loss function and optimizer / 定義損失函數與優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# 8️⃣ Training loop for multiple epochs / 執行多個 epoch 的訓練迴圈
for epoch in range(config.epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        # move to device / 移至裝置（GPU/CPU）
        images, labels = images.to(device), labels.to(device)

        # Forward pass / 前向傳播
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        # Backward pass / 反向傳播與參數更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加 Loss 與正確預測數
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(predictions, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # compute epoch metrics / 計算當前 epoch 的平均 Loss 與 Accuracy
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"Epoch [{epoch+1}/{config.epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")

    # log {"loss": ..., "accuracy": ...} to wandb / 將 loss 與 accuracy 記錄至 wandb
    wandb.log({
        "epoch": epoch + 1,
        "loss": epoch_loss,
        "accuracy": epoch_acc
    })

# 9️⃣ Save model parameters / 儲存模型參數
torch.save(model.state_dict(), "model.pt")
print("Model saved to model.pt")

# 結束 wandb 紀錄
wandb.finish()