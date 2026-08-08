# 1️⃣ Import necessary libraries
import os
import time
import torch
import torch.nn
import torch.optim
import torchvision
from torchvision import datasets, transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import wandb
# print(FashionMNIST)
# 2️⃣ Initialize wandb with project name and config
wandb.init(project="mnist-training",name=f"{os.getenv('USER','student')}_run_{int(time.time())}", config={"epochs": 5,"batch_size": 64,"lr": 1e-3})
config=wandb.config

# 3️⃣ Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4️⃣ Load dataset with transform
dataset = datasets.FashionMNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)

# 5️⃣ Create DataLoader for batch processing
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 6️⃣ Define MLP model
class MLP(torch.nn.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        layers = [torch.nn.Flatten()]
        input_size = 28 * 28
        for hidden_size in hidden_layers:
            layers.append(torch.nn.Linear(input_size, hidden_size))
            layers.append(torch.nn.ReLU())
            input_size = hidden_size
        layers.append(torch.nn.Linear(input_size, 10))
        self.layers = torch.nn.Sequential(*layers)
    def forward(self, x):
        output=self.layers(x)
        return output

model = MLP(hidden_layers=[256, 128]).to(device)

# 7️⃣ Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 8️⃣ Training loop for multiple epochs
for epoch in range(config.epochs):
    running_loss = 0
    correct_predictions = 0
    total_samples = 0
    for batch in loader:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct_predictions += (predictions.argmax(1) == labels).sum().item()
        total_samples += labels.size(0)

    loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    wandb.log({"loss": loss, "accuracy": accuracy})

# 9️⃣ Save model parameters
torch.save(model.state_dict(), "model.pt")
