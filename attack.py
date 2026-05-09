# attack.py
# CTEC 450 - AI Security Project
# Step 2: Fast Gradient Sign Method (FGSM) Attack

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from model import MnistCNN  # reuse the architecture

# ── Config ───────────────────────────────────────────────────────
MODEL_PATH  = "mnist_cnn.pth"
BATCH_SIZE  = 64
EPSILONS    = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # attack strengths

# ── Data ─────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ── FGSM Attack ───────────────────────────────────────────────────
def fgsm_attack(image, epsilon, gradient):
    """
    Perturb the image by stepping epsilon in the direction of the gradient sign.
    x_adv = x + epsilon * sign(∇_x J(θ, x, y))
    """
    perturbation = epsilon * gradient.sign()
    adversarial  = image + perturbation
    # Clamp to keep image in valid range after normalization
    adversarial  = torch.clamp(adversarial, -1, 1)
    return adversarial

# ── Evaluate Under Attack ─────────────────────────────────────────
def evaluate_attack(model, loader, epsilon, criterion, device):
    model.eval()
    correct = 0
    adv_examples = []  # store a few for visualization

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images.requires_grad = True

        # Forward pass with clean images
        outputs = model(images)
        loss    = criterion(outputs, labels)

        # Backward pass to get gradients w.r.t. input
        model.zero_grad()
        loss.backward()
        grad = images.grad.data

        # Generate adversarial examples
        adv_images = fgsm_attack(images, epsilon, grad)

        # Re-evaluate on adversarial examples
        with torch.no_grad():
            adv_outputs = model(adv_images)
        preds = adv_outputs.argmax(1)
        correct += (preds == labels).sum().item()

        # Save a few examples for visualization
        if len(adv_examples) < 5 and epsilon > 0:
            for i in range(min(5 - len(adv_examples), images.size(0))):
                if preds[i] != labels[i]:  # only save misclassified
                    adv_examples.append((
                        labels[i].item(),
                        preds[i].item(),
                        adv_images[i].detach().cpu().squeeze().numpy()
                    ))

    accuracy = correct / len(loader.dataset)
    return accuracy, adv_examples

# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MnistCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    criterion = nn.CrossEntropyLoss()

    print("Running FGSM attack across epsilon values...\n")
    accuracies     = []
    all_adv_examples = []

    for eps in EPSILONS:
        acc, examples = evaluate_attack(model, test_loader, eps, criterion, device)
        accuracies.append(acc)
        all_adv_examples.append(examples)
        print(f"  Epsilon: {eps:.2f}  →  Accuracy: {acc*100:.2f}%")

    # ── Plot: Accuracy vs Epsilon ─────────────────────────────────
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(EPSILONS, [a * 100 for a in accuracies], "o-", color="red")
    plt.xlabel("Epsilon (Attack Strength)")
    plt.ylabel("Accuracy (%)")
    plt.title("FGSM Attack – Model Accuracy vs Epsilon")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/fgsm_accuracy_drop.png")
    plt.show()
    print("\nGraph saved to results/fgsm_accuracy_drop.png")

    # ── Plot: Adversarial Example Samples ─────────────────────────
    # Show examples from epsilon=0.2 (index 4)
    sample_eps_index = 4
    examples = all_adv_examples[sample_eps_index]
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(12, 3))
        if len(examples) == 1:
            axes = [axes]
        for ax, (true_label, pred_label, img) in zip(axes, examples):
            ax.imshow(img, cmap="gray")
            ax.set_title(f"True: {true_label}\nPred: {pred_label}", color="red")
            ax.axis("off")
        fig.suptitle(f"Adversarial Examples (ε={EPSILONS[sample_eps_index]})", fontsize=13)
        plt.tight_layout()
        plt.savefig("results/adversarial_examples.png")
        plt.show()
        print("Adversarial examples saved to results/adversarial_examples.png")

    print(f"\nBaseline accuracy:         {accuracies[0]*100:.2f}%")
    print(f"Accuracy at ε=0.3 (max):   {accuracies[-1]*100:.2f}%")
    print(f"Accuracy drop:             {(accuracies[0]-accuracies[-1])*100:.2f}%")
