# 1️⃣ Import necessary libraries
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import wandb

# 2️⃣ Initialize wandb with project name and config
wandb.init(
    project="FashionMNIST_mlp",   # 專案名稱（W&B網頁上會出現）
    name="test_01",          # 這次實驗的名稱
    config={                      # 可選，方便記錄超參數
        "learning_rate": 0.001,   # learning_rate(lr):每次更新的步伐大小，常用值 0.001 ~ 0.0001
        "epochs": 5,
        "batch_size": 64
    }
)


# 3️⃣ Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4️⃣ Load dataset with transform
dataset = datasets.FashionMNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)

# 5️⃣ Create DataLoader for batch processing
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# 6️⃣ Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.layers(x)

model = MLP().to(device)

# 7️⃣ Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 8️⃣ Training loop for multiple epochs
for epoch in range(5):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (predictions.argmax(1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    wandb.log({
        "loss": avg_loss,
        "accuracy": accuracy
    })

# 9️⃣ Save model parameters
torch.save(model.state_dict(), "model.pt")