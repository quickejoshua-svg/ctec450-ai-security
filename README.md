# CTEC 450 â€“ AI Security Project
## Adversarial Attacks & Defenses on MNIST

---

## Project Structure

```
ctec450_ai_security/
â”œâ”€â”€ model.py          # Step 1: Train baseline CNN
â”œâ”€â”€ attack.py         # Step 2: FGSM adversarial attack
â”œâ”€â”€ defense.py        # Step 3: Adversarial training defense
â”œâ”€â”€ requirements.txt  # Python dependencies
â”œâ”€â”€ results/          # Auto-created: graphs & outputs
â”‚   â”œâ”€â”€ baseline_accuracy.png
â”‚   â”œâ”€â”€ fgsm_accuracy_drop.png
â”‚   â”œâ”€â”€ adversarial_examples.png
â”‚   â””â”€â”€ defense_comparison.png
â””â”€â”€ data/             # Auto-downloaded: MNIST dataset
```

---

## Setup (PyCharm)

1. Open PyCharm â†’ Open the `ctec450_ai_security/` folder
2. Open the terminal (bottom of PyCharm) and run:

```bash
pip install -r requirements.txt
```

---

## How to Run (in order)

### Step 1 â€“ Train Baseline Model
```bash
python model.py
```
- Downloads MNIST automatically
- Trains a CNN for 5 epochs
- Saves: `mnist_cnn.pth`, `results/baseline_accuracy.png`
- Expected test accuracy: ~99%

---

### Step 2 â€“ Run FGSM Attack
```bash
python attack.py
```
- Loads `mnist_cnn.pth`
- Runs FGSM at epsilon values: 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
- Saves: `results/fgsm_accuracy_drop.png`, `results/adversarial_examples.png`
- Expected: accuracy drops sharply as epsilon increases

---

### Step 3 â€“ Run Adversarial Defense
```bash
python defense.py
```
- Loads `mnist_cnn.pth` as a starting point
- Retrains using adversarial training (clean + FGSM examples per batch)
- Saves: `mnist_cnn_defended.pth`, `results/defense_comparison.png`
- Expected: defended model holds significantly better accuracy under attack

---

## What Each File Does

| File | Purpose |
|------|---------|
| `model.py` | Defines `MnistCNN` architecture, trains it, saves weights |
| `attack.py` | Implements `fgsm_attack()`, evaluates across epsilon values |
| `defense.py` | Adversarial training loop, compares original vs defended model |

---

## Expected Results Summary

| Condition | Accuracy |
|-----------|----------|
| Clean (no attack) | ~99% |
| FGSM Îµ=0.1 (no defense) | ~50â€“70% |
| FGSM Îµ=0.3 (no defense) | ~5â€“15% |
| FGSM Îµ=0.3 (with defense) | ~50â€“70% |

Exact numbers will vary slightly per run.

