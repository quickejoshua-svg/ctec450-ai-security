# model.py
# CTEC 450 - AI Security Project
# Step 1: Build and train a baseline CNN on MNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ── Reproducibility ──────────────────────────────────────────────
torch.manual_seed(42)

# ── Hyperparameters ──────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 5
LEARNING_RATE = 0.001
SAVE_PATH   = "mnist_cnn.pth"

# ── Data Loading ─────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),               # [0,255] → [0.0, 1.0]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean & std
])

train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── CNN Architecture ─────────────────────────────────────────────
class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 14x14 → 7x7
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc_block(x)
        return x

# ── Training Function ────────────────────────────────────────────
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ── Evaluation Function ──────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model     = MnistCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_accs, test_accs = [], []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        train_accs.append(tr_acc)
        test_accs.append(te_acc)
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc*100:.2f}% | "
              f"Test Acc: {te_acc*100:.2f}%")

    # Save model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), [a * 100 for a in train_accs], label="Train Accuracy")
    plt.plot(range(1, EPOCHS + 1), [a * 100 for a in test_accs],  label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Baseline Model – Training & Test Accuracy")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/baseline_accuracy.png")
    plt.show()
    print("Graph saved to results/baseline_accuracy.png")
