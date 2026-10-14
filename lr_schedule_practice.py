# -*- coding: utf-8 -*-
"""Compare a fixed learning rate with StepLR on one trainable parameter.

Run with the Python interpreter configured in your IDE (with PyTorch installed).
"""

import torch


def run_experiment(name, use_scheduler):
    weight = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([weight], lr=1.05)
    scheduler = None
    if use_scheduler:
        # Halve the learning rate after every 3 calls to scheduler.step().
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.5
        )

    print(f"\n{name}")
    print("step | lr used | weight after update | loss after update")
    for step in range(1, 10):
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.zero_grad()
        loss = (weight - 2.0).square()
        loss.backward()
        optimizer.step()

        # The scheduler sets the learning rate for the *next* update.
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            new_loss = (weight - 2.0).square().item()
            print(f"{step:>4} | {lr_used:>7.4f} | "
                  f"{weight.item():>19.4f} | {new_loss:>17.4f}")


run_experiment("Fixed learning rate", use_scheduler=False)
run_experiment("StepLR: halve every 3 updates", use_scheduler=True)
