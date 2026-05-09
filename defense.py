# defense.py
# CTEC 450 - AI Security Project
# Step 3: Defense via Adversarial Training

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from model  import MnistCNN
from attack import fgsm_attack, evaluate_attack, EPSILONS

# ── Config ───────────────────────────────────────────────────────
EPOCHS        = 5
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
ADV_EPSILON   = 0.2   # epsilon used during adversarial training
SAVE_PATH     = "mnist_cnn_defended.pth"

# ── Data ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ── Adversarial Training Loop ─────────────────────────────────────
# For each batch: generate adversarial examples on-the-fly using FGSM,
# then train on a 50/50 mix of clean + adversarial images.
def train_adversarial(model, loader, optimizer, criterion, epsilon, device):
    model.train()
    total_loss, correct = 0.0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        # ── Generate adversarial examples ──────────────────────────
        images.requires_grad = True
        outputs = model(images)
        loss    = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad        = images.grad.data
        adv_images  = fgsm_attack(images, epsilon, grad)
        images      = images.detach()

        # ── Mix clean + adversarial ────────────────────────────────
        mixed_images = torch.cat([images, adv_images], dim=0)
        mixed_labels = torch.cat([labels, labels],    dim=0)

        # ── Train on mixed batch ───────────────────────────────────
        optimizer.zero_grad()
        mixed_outputs = model(mixed_images)
        mixed_loss    = criterion(mixed_outputs, mixed_labels)
        mixed_loss.backward()
        optimizer.step()

        total_loss += mixed_loss.item() * mixed_images.size(0)
        correct    += (mixed_outputs.argmax(1) == mixed_labels).sum().item()

    return total_loss / (2 * len(loader.dataset)), correct / (2 * len(loader.dataset))

def evaluate_clean(model, loader, criterion, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            correct += (model(images).argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # ── Load original model weights as starting point ─────────────
    model     = MnistCNN().to(device)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Adversarial training with ε={ADV_EPSILON}\n")
    clean_accs = []

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_adversarial(model, train_loader, optimizer, criterion, ADV_EPSILON, device)
        clean_acc       = evaluate_clean(model, test_loader, criterion, device)
        clean_accs.append(clean_acc)
        print(f"Epoch {epoch}/{EPOCHS} | Train Acc (mixed): {tr_acc*100:.2f}% | "
              f"Clean Test Acc: {clean_acc*100:.2f}%")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nDefended model saved to {SAVE_PATH}")

    # ── Compare: Original vs Defended across all epsilons ─────────
    print("\nEvaluating BOTH models across epsilon values...\n")

    original_model = MnistCNN().to(device)
    original_model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))

    orig_accs, def_accs = [], []
    for eps in EPSILONS:
        orig_acc, _ = evaluate_attack(original_model, test_loader, eps, criterion, device)
        def_acc,  _ = evaluate_attack(model,          test_loader, eps, criterion, device)
        orig_accs.append(orig_acc)
        def_accs.append(def_acc)
        print(f"  ε={eps:.2f}  | Original: {orig_acc*100:.2f}%  | Defended: {def_acc*100:.2f}%")

    # ── Plot comparison ────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(EPSILONS, [a * 100 for a in orig_accs], "o-r", label="Original (no defense)")
    plt.plot(EPSILONS, [a * 100 for a in def_accs],  "o-g", label="Adversarially Trained")
    plt.xlabel("Epsilon (Attack Strength)")
    plt.ylabel("Accuracy (%)")
    plt.title("Defense Comparison – Original vs Adversarially Trained Model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/defense_comparison.png")
    plt.show()
    print("\nComparison graph saved to results/defense_comparison.png")

    print(f"\n── Summary ────────────────────────────────────────")
    print(f"At ε=0.3 (strongest attack):")
    print(f"  Original model accuracy:  {orig_accs[-1]*100:.2f}%")
    print(f"  Defended model accuracy:  {def_accs[-1]*100:.2f}%")
    print(f"  Improvement:              +{(def_accs[-1]-orig_accs[-1])*100:.2f}%")
