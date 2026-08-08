# Week 1 – MNIST / FashionMNIST MLP Training

**Student:** Manaswi Patil  
**Course:** VISLab Summer Training  
**Task:** Implement an MLP for FashionMNIST using PyTorch and track experiments with Weights & Biases.

---

## Model
A simple Multi-Layer Perceptron (MLP):

Input: 28×28 image  
Flatten → Linear(784 → 256) → ReLU  
→ Linear(256 → 128) → ReLU  
→ Linear(128 → 10)

Loss Function: CrossEntropyLoss  
Optimizer: Adam  
Epochs: 5  
Batch Size: 64

---

## Results
Final Training Accuracy: **~89%**

Training metrics were tracked using **Weights & Biases**.

Run link: https://wandb.ai/msw-ai-national-chung-cheng-university/week1_mlp?nw=nwusermswai
