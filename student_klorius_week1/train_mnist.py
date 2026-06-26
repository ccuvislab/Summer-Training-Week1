import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.optim import Adam
import wandb

# 2️⃣ 初始化 wandb
wandb.init(project="summer-training-week1", name="mlp-fashionmnist")

# 3️⃣ 設定裝置 (GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的裝置為：{device}")

# 4️⃣ 載入資料集 (增加 Normalize 有助於模型收斂)
transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)

# 5️⃣ 建立 DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True)

# 6️⃣ 定義 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.layers(self.flatten(x))

model = MLP().to(device)

# 7️⃣ 定義損失函數與優化器
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 8️⃣ 訓練迴圈
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        import os
        # 🔥 自動記錄 GPU 狀態至 Log 檔
        if epoch == 0 and batch_idx == 0:
            print("\n" + "="*50)
            print("📸 自動呼叫 nvidia-smi 記錄 GPU 狀態：")
            os.system("nvidia-smi")
            print("="*50 + "\n")
            
        # 前向傳播
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 統計
        running_loss += loss.item()
        _, predicted = torch.max(predictions.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 計算並輸出結果
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    
    # 紀錄至 wandb
    wandb.log({"epoch": epoch+1, "loss": epoch_loss, "accuracy": epoch_acc})

# 9️⃣ 儲存模型
torch.save(model.state_dict(), "model.pt")
print("訓練結束，模型已儲存為 model.pt")


