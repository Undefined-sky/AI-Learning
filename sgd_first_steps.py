# -*- coding: utf-8 -*-
"""A first optimizer experiment: watch SGD update one linear layer."""

import torch
from torch import nn


# Two samples. Each inner list is one sample with one input feature.
x = torch.tensor([[1.0], [2.0]])
target = 2.0 * x + 3.0  # [[5.0], [7.0]]

model = nn.Linear(1, 1)  # prediction = weight * x + bias
with torch.no_grad():
    model.weight.fill_(0.0)
    model.bias.fill_(0.0)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print("x:", x.tolist(), "shape:", tuple(x.shape))
print("target:", target.tolist(), "shape:", tuple(target.shape))
print("initial weight:", model.weight.item(), "bias:", model.bias.item())

for step in range(1, 4):
    optimizer.zero_grad()          # Clear the previous round's gradients.
    prediction = model(x)         # Use the current weight and bias.
    loss = criterion(prediction, target)
    loss.backward()               # Fill weight.grad and bias.grad.

    print(f"\nstep {step}, BEFORE optimizer.step()")
    print("  prediction:", prediction.detach().flatten().tolist())
    print("  loss:", round(loss.item(), 6))
    print("  weight.grad:", round(model.weight.grad.item(), 6))
    print("  bias.grad:", round(model.bias.grad.item(), 6))

    optimizer.step()               # Actually change weight and bias.
    with torch.no_grad():
        new_loss = criterion(model(x), target)

    print(f"step {step}, AFTER optimizer.step()")
    print("  weight:", round(model.weight.item(), 6))
    print("  bias:", round(model.bias.item(), 6))
    print("  new loss:", round(new_loss.item(), 6))
