# -*- coding: utf-8 -*-
"""Practice saving and resuming a model, optimizer, and LR scheduler.

Run with the PyTorch-enabled Python interpreter in your IDE.
"""

from pathlib import Path

import torch
from torch import nn


checkpoint_path = Path(__file__).with_name("checkpoint_practice.pt")
x = torch.tensor([[1.0], [2.0]])# 两个样本，每个样本有一个输入特征
target = torch.tensor([[5.0], [7.0]])#目标张量
criterion = nn.MSELoss()#选用了MSE损失函数模型


def build_training_parts():
    model = nn.Linear(1, 1)# 一个输入特征、一个输出特征的线性层
    with torch.no_grad():#初始化weight和bias为0
        model.weight.zero_()
        model.bias.zero_()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)# AdamW：学习率 0.1，权重衰减系数 0.01
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5#学习率衰减是三轮一次，每次衰减1/2
    )
    return model, optimizer, scheduler


def train_one_epoch(epoch, model, optimizer, scheduler):
    lr_used = optimizer.param_groups[0]["lr"]# 读取本轮训练使用的学习率
    optimizer.zero_grad()#梯度清零
    prediction = model(x)#预测值
    loss = criterion(prediction, target)#mse损失计算
    loss.backward()#反向传播
    optimizer.step()#优化权重bias
    scheduler.step()  # Once per epoch: set the learning rate for the next epoch.#更新学习率
    print(f"epoch {epoch}: lr used={lr_used:.3f}, loss before update={loss.item():.4f}")


print("Train for four epochs, then save a checkpoint:")
model, optimizer, scheduler = build_training_parts()
for epoch in range(1, 5):
    train_one_epoch(epoch, model, optimizer, scheduler)# 先训练第 1～4 轮

torch.save(#保存了训练结果，模型的情况，优化器的情况，
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),#优化器
        "scheduler": scheduler.state_dict(),#学习率衰减情况
        "next_epoch": 5,
    },
    checkpoint_path,# 检查点文件的保存位置
)
print(f"Saved: {checkpoint_path.name}")


print("\nCreate fresh objects, restore, and finish epochs five and six:")
resumed_model, resumed_optimizer, resumed_scheduler = build_training_parts()# 创建新的对象；此时尚未恢复训练状态
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)# 将文件中的状态字典读入内存
resumed_model.load_state_dict(checkpoint["model"])
resumed_scheduler.load_state_dict(checkpoint["scheduler"])# 恢复调度器的进度
resumed_optimizer.load_state_dict(checkpoint["optimizer"])

for epoch in range(checkpoint["next_epoch"], 7):
    train_one_epoch(epoch, resumed_model, resumed_optimizer, resumed_scheduler)


# A continuous six-epoch run should reach the same parameters in this
# deterministic, full-batch example.
continuous_model, continuous_optimizer, continuous_scheduler = build_training_parts()# 连续训练 6 轮，作为恢复训练的对照
for epoch in range(1, 7):
    continuous_optimizer.zero_grad()
    loss = criterion(continuous_model(x), target)
    loss.backward()
    continuous_optimizer.step()
    continuous_scheduler.step()

torch.testing.assert_close(resumed_model.weight, continuous_model.weight)
torch.testing.assert_close(resumed_model.bias, continuous_model.bias)
print("\nCheck passed: resumed and uninterrupted training match.")
