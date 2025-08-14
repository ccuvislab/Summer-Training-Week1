import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb

def main():
    # 2️⃣ Initialize wandb
    wandb.init(project="fashionmnist-mlp", config={
        "epochs": 5,
        "batch_size": 64,
        "learning_rate": 0.001,
        "hidden_units": 256
    })
    config = wandb.config

    # 3️⃣ Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4️⃣ Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=transform, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 6️⃣ Model
    class MLP(nn.Module):
        def __init__(self, input_size=28*28, hidden_units=256, num_classes=10):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_units)
            self.fc2 = nn.Linear(hidden_units, hidden_units)
            self.fc3 = nn.Linear(hidden_units, num_classes)

        def forward(self, x):
            x = x.view(-1, 28*28)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = MLP(hidden_units=config.hidden_units).to(device)

    # 7️⃣ Loss & Optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 8️⃣ Training
    for epoch in range(config.epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = predictions.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total

        # Testing
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_acc = 100. * test_correct / test_total

        # Log
        wandb.log({
            "epoch": epoch+1,
            "train_loss": total_loss / len(train_loader),
            "train_acc": train_acc,
            "test_loss": test_loss / len(test_loader),
            "test_acc": test_acc
        })

        print(f"Epoch [{epoch+1}/{config.epochs}] "
              f"Train Loss: {total_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% "
              f"Test Loss: {test_loss / len(test_loader):.4f} | Test Acc: {test_acc:.2f}%")

    # 9️⃣ Save model
    torch.save(model.state_dict(), "model.pt")
    wandb.save("model.pt")

if __name__ == "__main__":
    main()
