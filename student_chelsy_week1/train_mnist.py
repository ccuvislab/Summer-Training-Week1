# 1️⃣ Import necessary libraries
import torch
import torchvision
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.data import DataLoader

# 2️⃣ Initialize wandb with project name and config
wandb.init(project="summer_hw1_mnist", name=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",dir="week1_wandb", config={"epochs":5,"learning_rate":0.001,"batch_size":64})
config = wandb.config
lr=config.learning_rate
num_epochs=config.epochs
batch_size=config.batch_size
# 3️⃣ Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

# 4️⃣ Load dataset with transform
dataset = torchvision.datasets.MNIST(root="./mnist",train=True, transform=transforms.ToTensor(), download=True)

# 5️⃣ Create DataLoader for batch processing
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 6️⃣ Define MLP model
class MLP(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super().__init__()
        self.fc1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.fc2=nn.Linear(hidden_size,num_classes)
    def forward(self, x):
        # print(x.size())#[64,1,28,28]
        x = x.view(x.size(0), -1)#原本是(batch_size,1,28,28)，view成(batch_size,784)
        out=self.fc1(x)
        out=self.relu(out)
        output=self.fc2(out)
        return output

model = MLP(input_size=28*28,hidden_size=128,num_classes=10).to(device)

# 7️⃣ Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 8️⃣ Training loop for multiple epochs
for epoch in range(num_epochs):
    total_correct=0
    total=0
    for batch in loader:
        images, labels = batch
        images=images.to(device)
        labels=labels.to(device)
        predictions = model(images)
        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(predictions, 1)#predictions是10個類別的機率
        total_correct += (predicted == labels).sum().item()
        total+=labels.size(0)

    accuracy=total_correct/total
    wandb.log ({"loss": loss, "accuracy":accuracy})

# 9️⃣ Save model parameters
torch.save(model.state_dict(), "model.pt")
