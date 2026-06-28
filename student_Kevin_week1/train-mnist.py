# 1️⃣ Import necessary libraries / 載入必要的套件
import torch
import torchvision
import wandb
from torchvision import transforms
from torch.utils.data import DataLoader

# 2️⃣ Initialize wandb with project name and config / 初始化 wandb，設定專案名稱與超參數設定
num_epochs = 5
wandb.init(project="summer_train_week1", config={"epochs": num_epochs, "batch_size": 64, "learning_rate": 0.001})

# 3️⃣ Set device (GPU if available) / 設定裝置（優先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4️⃣ Load dataset with transform / 載入資料集並套用轉換（transform）
dataset = torchvision.datasets.FashionMNIST(root = "./data", train = True, transform=transforms.ToTensor(), download=True)

# 5️⃣ Create DataLoader for batch processing / 建立 DataLoader 進行批次處理
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 6️⃣ Define MLP model / 定義 MLP 模型
class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)
        # define layers... / 定義各層...
    def forward(self, x):
        x = x.view(-1, 784)

        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = MLP().to(device)

# 7️⃣ Define loss function and optimizer / 定義損失函數與優化器
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
# 8️⃣ Training loop for multiple epochs / 執行多個 epoch 的訓練迴圈
for epoch in range(num_epochs):
    total = 0
    total_loss = 0
    correct = 0
    for batch in loader:
        images, labels = batch
        # move to device / 移至裝置（GPU/CPU）
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += labels.size(0)
        total_loss += loss.item()
        _, pred = torch.max(predictions, 1)
        correct += (pred == labels).sum().item()

    # compute accuracy / 計算準確率
    # log {"loss": ..., "accuracy": ...} to wandb / 將 loss 與 accuracy 記錄至 wandb
    
    accuracy = correct / total
    wandb.log({"loss": total_loss / len(loader), "accuracy": accuracy})

# 9️⃣ Save model parameters / 儲存模型參數
torch.save(model.state_dict(), "model.pt")