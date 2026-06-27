import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

# 1 設定超參數
config = {
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 10,
    "hidden_size": 128,
    "dataset": "FashionMNIST",
    "model": "MLP"
}

# 2 初始化 wandb
wandb.init(
    project="summer_hw1",
    config=config
)

# 3 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 4 資料前處理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 5 載入 FashionMNIST 訓練資料
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True
)

# 6 定義 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, config["hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["hidden_size"], 10)
        )

    def forward(self, x):
        return self.model(x)

model = MLP().to(device)

# 7 定義 loss function 和 optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config["learning_rate"]
)

# 8 執行10個epoch的訓練迴圈
for epoch in range(config["num_epochs"]):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累積 loss
        total_loss += loss.item()

        # 計算 accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    print(
        f"Epoch [{epoch + 1}/{config['num_epochs']}], "
        f"Loss: {avg_loss:.4f}, "
        f"Accuracy: {accuracy:.4f}"
    )

    # 9 記錄到 wandb
    wandb.log(
        {
            "loss": avg_loss,
            "accuracy": accuracy
        },
        step=epoch + 1
    )

# 10 儲存模型參數
torch.save(model.state_dict(), "model.pt")
print("Model saved as model.pt")

# 11 結束 wandb
wandb.finish()
