# -*- coding: utf-8 -*-
"""AdamW practice: first isolate weight decay, then inspect validation loss.

Run with the Python interpreter configured in your IDE (with PyTorch installed).
Before running, predict the first two printed weights.
"""

import torch
from torch import nn


def one_step(optimizer_class, weight_decay):
    weight = nn.Parameter(torch.tensor([10.0]))
    optimizer = optimizer_class([weight], lr=0.1, weight_decay=weight_decay)

    optimizer.zero_grad()
    loss = (weight * 0).sum()  # Connected to weight, but gradient is exactly zero.
    loss.backward()
    optimizer.step()
    return weight.item(), weight.grad.item()


print("Part 1: initial weight = 10, learning rate = 0.1, gradient = 0")
for name, optimizer_class, decay in [
    ("AdamW, decay=0", torch.optim.AdamW, 0.0),
    ("AdamW, decay=0.2", torch.optim.AdamW, 0.2),
    ("Adam,  decay=0.2", torch.optim.Adam, 0.2),
]:
    weight, grad = one_step(optimizer_class, decay)
    print(f"{name:18s} grad={grad:.1f}, weight after step={weight:.4f}")


print("\nPart 2: fit noisy training data; compare clean validation data")
train_x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
train_y = torch.tensor([[2.0], [4.0], [6.0], [11.0]])  # Last label has noise.
val_x = torch.tensor([[5.0], [6.0]])
val_y = torch.tensor([[10.0], [12.0]])
criterion = nn.MSELoss()


def train(weight_decay):
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05,
                                  weight_decay=weight_decay)
    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_loss = criterion(model(train_x), train_y).item()
        val_loss = criterion(model(val_x), val_y).item()
        return model.weight.item(), train_loss, val_loss


for decay in (0.0, 0.2):
    weight, train_loss, val_loss = train(decay)
    print(f"decay={decay:.1f}: weight={weight:.4f}, "
          f"train loss={train_loss:.4f}, val loss={val_loss:.4f}")
