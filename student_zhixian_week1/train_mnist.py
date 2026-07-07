# 1️⃣ Import necessary libraries / 載入必要的套件
import torch
import torch.nn as nn                         
from torch.utils.data import DataLoader      
from torch.optim import Adam                 
import torchvision
import torchvision.transforms as transforms
import wandb

# 2️⃣ Initialize wandb with project name and config / 初始化 wandb，設定專案名稱與超參數設定
wandb.init(project="MNIST_MLP", config={"learning_rate": 0.001, "epochs": 5, "batch_size": 64})

# 3️⃣ Set device (GPU if available) / 設定裝置（優先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4️⃣ Load dataset with transform / 載入資料集並套用轉換（transform）
dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)

# 5️⃣ Create DataLoader for batch processing / 建立 DataLoader 進行批次處理
loader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)

# 6️⃣ Define MLP model / 定義 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # define layers... / 定義各層...
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10) # 10 個數字分類
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        return output

model = MLP().to(device)

# 7️⃣ Define loss function and optimizer / 定義損失函數與優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=wandb.config.learning_rate)

# 8️⃣ Training loop for multiple epochs / 執行多個 epoch 的訓練迴圈
for epoch in range(wandb.config.epochs):
    model.train()
    
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        images, labels = batch
        # move to device / 移至裝置（GPU/CPU）
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # compute accuracy / 計算準確率
    epoch_loss = total_loss / total
    epoch_acc = correct / total

    # log {"loss": ..., "accuracy": ...} to wandb / 將 loss 與 accuracy 記錄至 wandb
    wandb.log({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_acc})

# 9️⃣ Save model parameters / 儲存模型參數
torch.save(model.state_dict(), "model.pt")