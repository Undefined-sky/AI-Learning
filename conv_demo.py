# -*- coding: utf-8 -*-
"""
小土堆《土堆说卷积操作》配套实验

跑法：在 PyCharm 里右键这个文件 -> Run 'conv_demo'

四个部分：
  Part 1  手算验证 —— 一次卷积到底算了什么
  Part 2  stride / padding 分别改变了什么
  Part 3  真网络里的形状追踪 —— 解释 nn.Linear 的 in_features 从哪来
  Part 4  多通道卷积 —— 为什么会报"通道数不匹配"

看的时候重点看三样东西：shape、值、和你手算的答案对不对得上。
"""

import torch
import torch.nn.functional as F
from torch import nn


def show(title, t):
    """统一的打印格式：先报形状，再报值"""
    print(f"\n{'-' * 46}")
    print(title)
    print(f"  shape: {tuple(t.shape)}")
    print(f"  值:\n{t.squeeze()}")


# ============================================================
print("=" * 46)
print("Part 1  手算验证：一次卷积到底算了什么")
print("=" * 46)
# ============================================================

# 造一个 5x5 的"图"和一个 3x3 的"卷积核"
# 注意：这里的数字是我们手写的，不是学出来的
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]], dtype=torch.float32)

show("(1) 原始 input（2 维）", input)
show("(2) 原始 kernel（2 维）", kernel)

# conv2d 只认 4 维：(batch, channel, H, W)
# 这里的 batch=1, channel=1 是我们硬凑出来的，为了满足函数要求
input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

show("(3) reshape 后的 input（4 维）", input)
show("(4) reshape 后的 kernel（4 维）", kernel)

# ---- 默认 stride=1 ----
output = F.conv2d(input, kernel, stride=1)
show("(5) stride=1 的输出", output)

print("\n  对照手算结果：")
print("    10 12 12")
print("    18 16 16")
print("    13  9  3")

# ---- 亲手拆开左上角那个 10 ----
patch = input[0, 0, 0:3, 0:3]
print("\n  [核对] 左上角 10 的来历：")
print(f"    窗口盖住的 3x3:\n{patch}")
print(f"    卷积核 3x3:\n{kernel[0, 0]}")
print(f"    逐位相乘再求和 = {float((patch * kernel[0, 0]).sum())}")


# ============================================================
print("\n\n" + "=" * 46)
print("Part 2  stride / padding 分别改变了什么")
print("=" * 46)
# ============================================================

out_s1 = F.conv2d(input, kernel, stride=1)
show("(1) stride=1, padding=0", out_s1)

out_s2 = F.conv2d(input, kernel, stride=2)
show("(2) stride=2, padding=0  <- 窗口一次跨 2 格", out_s2)

out_p1 = F.conv2d(input, kernel, stride=1, padding=1)
show("(3) stride=1, padding=1  <- 四周补了一圈 0", out_p1)


def out_size(h_in, k, stride=1, padding=0):
    """输出尺寸公式：H_out = (H_in + 2*padding - k) / stride + 1"""
    return (h_in + 2 * padding - k) // stride + 1


print("\n  公式验证：  H_out = (H_in + 2*padding - k) // stride + 1")
for name, s, p, actual in [("stride=1,padding=0", 1, 0, out_s1),
                           ("stride=2,padding=0", 2, 0, out_s2),
                           ("stride=1,padding=1", 1, 1, out_p1)]:
    print(f"    {name:20s} 公式预测 {out_size(5, 3, s, p)}"
          f"   实际 {actual.shape[-1]}")

print("\n  记住两句话就够：")
print("    stride 大  -> 滑得少  -> 输出变小")
print("    padding 大 -> 图变大  -> 输出变大（且边缘不再被忽略）")


# ============================================================
print("\n\n" + "=" * 46)
print("Part 3  真网络里的形状追踪")
print("=" * 46)
# ============================================================
# 说明：下面这些层本该写在 __init__ 里的 self.xxx = ...
#       这里为了看清楚形状变化，直接一行建一行用
#
# 这个结构就是小土堆"搭建小实战"那一章的 Tudui

x = torch.randn(2, 3, 32, 32)        # 假装是 2 张 CIFAR 图片
print(f"\n  输入                    {tuple(x.shape)}")

x = nn.Conv2d(3, 32, 5, padding=2)(x)
print(f"  Conv2d(3, 32, 5, p=2)   {tuple(x.shape)}   <- 通道 3->32，尺寸不变")

x = nn.MaxPool2d(2)(x)
print(f"  MaxPool2d(2)            {tuple(x.shape)}   <- 尺寸减半")

x = nn.Conv2d(32, 32, 5, padding=2)(x)
print(f"  Conv2d(32, 32, 5, p=2)  {tuple(x.shape)}")

x = nn.MaxPool2d(2)(x)
print(f"  MaxPool2d(2)            {tuple(x.shape)}")

x = nn.Conv2d(32, 64, 5, padding=2)(x)
print(f"  Conv2d(32, 64, 5, p=2)  {tuple(x.shape)}   <- 通道 32->64")

x = nn.MaxPool2d(2)(x)
print(f"  MaxPool2d(2)            {tuple(x.shape)}")

x = x.flatten(1)
print(f"  flatten(1)              {tuple(x.shape)}   <== 64*4*4 = 1024")

x = nn.Linear(1024, 64)(x)
print(f"  Linear(1024, 64)        {tuple(x.shape)}")

x = nn.Linear(64, 10)(x)
print(f"  Linear(64, 10)          {tuple(x.shape)}   <- 10 分类")

print("\n  看到那个 1024 了吗？它不是凭空来的，")
print("  是把前面每一步的形状算下来，最后 64*4*4 得到的。")
print("  改任何一个参数，这个数字都要重算 —— 这就是初学者最常卡的地方。")


# ============================================================
print("\n\n" + "=" * 46)
print("Part 4  多通道卷积：为什么会报通道数不匹配")
print("=" * 46)
# ============================================================

x = torch.randn(1, 3, 4, 4)          # 1 张 3 通道小图
w_ok = torch.randn(2, 3, 3, 3)       # 2 个卷积核，每个 3 层厚
print(f"\n  x  = {tuple(x.shape)}   (batch=1, 通道=3, 4x4)")
print(f"  w  = {tuple(w_ok.shape)}   (输出通道=2, 输入通道=3, 3x3)")
print(f"  通道匹配 -> 输出 {tuple(F.conv2d(x, w_ok).shape)}")

w_bad = torch.randn(2, 1, 3, 3)      # 只做了 1 层厚
print(f"\n  w  = {tuple(w_bad.shape)}   (输出通道=2, 输入通道=1, 3x3)")
try:
    F.conv2d(x, w_bad)
except RuntimeError as e:
    print(f"  通道不匹配 -> 报错：{str(e).splitlines()[0]}")

print("\n  铁律：卷积核的'输入通道'必须等于输入的通道数。")
print("  灰度图 1 通道、彩色图 3 通道 —— 所以彩色图第一层永远是 Conv2d(3, ...)。")
