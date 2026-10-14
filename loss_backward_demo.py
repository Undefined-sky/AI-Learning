# -*- coding: utf-8 -*-
"""
损失函数与反向传播：最小可观察实验

目标：把一次训练拆成四步，并观察每一步到底改变了什么：
    预测 -> 算损失 -> backward() 算梯度 -> step() 更新参数

运行：在 PyCharm 中右键运行本文件。
"""

import torch
from torch import nn


def show_state(title, model, x, target):
    """统一打印：参数、预测值和当前损失。"""
    with torch.no_grad():
        prediction = model(x)
        loss = nn.MSELoss()(prediction, target)

    print(f"\n{'=' * 56}\n{title}")
    print(f"weight = {model.weight.item():.2f}")
    print(f"bias   = {model.bias.item():.2f}")
    print(f"预测 y = {prediction.item():.2f}")
    print(f"目标值 = {target.item():.2f}")
    print(f"当前 loss = (预测 - 目标)^2 = {loss.item():.2f}")


# 一个只有一个权重、一个 bias 的模型：y = w*x + b
model = nn.Linear(1, 1)
with torch.no_grad():
    model.weight.fill_(1.0)
    model.bias.fill_(0.0)

x = torch.tensor([[2.0]])
target = torch.tensor([[5.0]])
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

show_state("第 0 步：刚初始化，参数还是随机前设定的 w=1, b=0", model, x, target)

# ------------------------- 1. 前向传播 + 损失 -------------------------
prediction = model(x)
loss = criterion(prediction, target)

print("\n第 1 步：前向传播并计算损失")
print(f"  model(x) = {prediction.item():.2f}")
print(f"  loss = {loss.item():.2f}")
print(f"  此时 weight.grad = {model.weight.grad}")
print("  只有 prediction 和 loss 被算出来；参数和梯度都还没有变化。")

# ----------------------------- 2. 反向传播 -----------------------------
loss.backward()

print("\n第 2 步：loss.backward()")
print(f"  weight 仍是 {model.weight.item():.2f}  <- 参数没有更新")
print(f"  bias   仍是 {model.bias.item():.2f}  <- 参数没有更新")
print(f"  weight.grad = {model.weight.grad.item():.2f}")
print(f"  bias.grad   = {model.bias.grad.item():.2f}")
print("  grad 为负，表示把参数调大一点，会让 loss 下降。")

# ----------------------------- 3. 参数更新 -----------------------------
optimizer.step()

print("\n第 3 步：optimizer.step()")
print("  优化器执行：参数 = 参数 - 学习率 * 参数.grad")
print(f"  新 weight = {model.weight.item():.2f}")
print(f"  新 bias   = {model.bias.item():.2f}")

show_state("第 4 步：用更新后的参数重新预测", model, x, target)

# ----------------------------- 4. 梯度清零 -----------------------------
optimizer.zero_grad()
print("\n第 5 步：optimizer.zero_grad()")
print(f"  weight.grad = {model.weight.grad}")
print(f"  bias.grad   = {model.bias.grad}")
print("  下一轮开始前清空旧梯度，避免新旧两轮的责任记录累加。")
