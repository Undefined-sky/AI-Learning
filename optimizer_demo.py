# -*- coding: utf-8 -*-
"""Optimizer practice: fit the simple relationship y = 2x + 3.

Run this file, compare the printed loss/parameters, then try changing the
learning rates or optimizer settings and predict what will happen first.
"""

import torch
from torch import nn


torch.manual_seed(0)

# A tiny, deterministic regression dataset: 21 examples, one feature each.
x = torch.linspace(-2.0, 2.0, 21).reshape(-1, 1)
target = 2.0 * x + 3.0
criterion = nn.MSELoss()


def make_model():
    """Use identical starting parameters for a fair optimizer comparison."""
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.0)
        model.bias.fill_(0.0)
    return model


def train_with(optimizer_name, optimizer_factory, steps=100):
    model = make_model()
    optimizer = optimizer_factory(model.parameters())
    with torch.no_grad():
        loss_records = {0: criterion(model(x), target).item()}
    report_steps = sorted({1, min(5, steps), min(20, steps), steps})

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        if step in report_steps:
            with torch.no_grad():
                loss_records[step] = criterion(model(x), target).item()

    loss_summary = "  ".join(
        f"loss@{step}={value:9.5f}" for step, value in loss_records.items()
    )
    print(
        f"{optimizer_name:16s} {loss_summary}  "
        f"w={model.weight.item():7.4f}  b={model.bias.item():7.4f}"
    )


print("Target relationship: y = 2x + 3")
print("All optimizers start with weight=0, bias=0, use the same data and lr=0.05.")
print("AdamW uses weight_decay=0.01, so its final parameters may be slightly shrunk.")
print()

train_with("SGD", lambda params: torch.optim.SGD(params, lr=0.05))
train_with(
    "SGD + Momentum",
    lambda params: torch.optim.SGD(params, lr=0.05, momentum=0.9),
)
train_with("Adam", lambda params: torch.optim.Adam(params, lr=0.05))
train_with(
    "AdamW",
    lambda params: torch.optim.AdamW(params, lr=0.05, weight_decay=0.01),
)


print("\nLearning-rate experiment: plain SGD, same model and data, 20 steps.")
train_with("SGD lr=0.05", lambda params: torch.optim.SGD(params, lr=0.05), steps=20)
train_with("SGD lr=0.80", lambda params: torch.optim.SGD(params, lr=0.80), steps=20)

print("\nTry next: change lr, momentum, or weight_decay, then predict the result before running.")
