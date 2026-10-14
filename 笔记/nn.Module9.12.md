# nn.Module 核心概念笔记（9.12）

## 一、它是什么
**PyTorch 官方的"神经网络标准模板"，所有网络都必须继承它。**

你只填两个空：

| 填哪里 | 填什么 |
|---|---|
| `__init__` | 网络**有哪些层** |
| `forward` | 数据**怎么流过**这些层 |

其余（管参数、搬 GPU、存模型、梯度清零）它全包了。

## 二、最小骨架
```python
import torch
from torch import nn

class Tudui(nn.Module):          # ① 继承 nn.Module
    def __init__(self):
        super().__init__()       # ② 必须第一行

    def forward(self, input):    # ③ 数据怎么走（名字固定不能改）
        output = input + 1
        return output

tudui = Tudui()
x = torch.tensor(1.0)
output = tudui(x)                # ④ 像调函数一样调
print(output)                    # tensor(2.)
```

## 三、四个必答问题

### ① `class Tudui(nn.Module)` = 继承
Tudui 自动拥有 nn.Module 的所有能力，还能加自己的。
> 比喻：nn.Module 是精装房的水电煤气管线，Tudui 是你的装修——你没装管线，但开水龙头就有水。

### ② 为什么必须 `super().__init__()`？
= 先让父类把准备工作做完（建立"参数账本"等）。
漏写 → `AttributeError: cannot assign module before Module.__init__() call`
> **口诀：必须是 `__init__` 的第一行。**

### ③ `forward` 是什么？
定义"数据从输入走到输出经过什么" = 一条流水线。
**名字固定是 `forward`，不能改。**

### ④ 为什么写 `tudui(x)` 而不是 `tudui.forward(x)`？
nn.Module 内部：
```python
class Module:
    def __call__(self, x):
        # 先做准备工作（hooks、检查）
        return self.forward(x)
```
| 写法 | 结果 |
|---|---|
| `tudui(x)` | ✅ 走 `__call__` → 准备 → forward |
| `tudui.forward(x)` | ⚠️ 跳过准备工作 |

**两种都出结果，但只有第一种正确。**
> 呼应 `__call__` 笔记：统一套路 `模块(数据)`，不用记各自方法名。

## 四、nn.Module 白送的能力（继承的真正理由）

| 能力 | 写法 | 干什么 |
|---|---|---|
| 拿到所有参数 | `tudui.parameters()` | **给优化器用**，训练必备 |
| 看参数名 | `tudui.named_parameters()` | 调试 |
| 搬到 GPU | `tudui.to("cuda")` | 网络+参数一起搬 |
| 训练/评估模式 | `tudui.train()` / `tudui.eval()` | 影响 Dropout、BatchNorm |
| 梯度清零 | `tudui.zero_grad()` | 训练循环每轮用 |
| 保存/读取 | `tudui.state_dict()` | 存成字典 |
| 看子模块 | `tudui.children()` | 调试 |
| 前向 | `tudui(x)` | 调你写的 forward |

## 五、真实网络骨架
```python
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        # 只"定义层"，不跑数据
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool  = nn.MaxPool2d(2)
        self.fc    = nn.Linear(6*15*15, 10)

    def forward(self, x):
        # 才"跑数据"
        x = self.conv1(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
```

### 细节 1：`__init__` 只定义，`forward` 才跑
- `__init__` 只在 `Tudui()` 那一刻跑**一次**
- `forward` 每次喂数据都跑

### 细节 2：层必须挂在 `self.` 上（⚠️ 最易犯的隐形错误）
```python
self.conv1 = nn.Conv2d(3, 6, 3)   # ✅ 被 nn.Module 登记
conv1 = nn.Conv2d(3, 6, 3)        # ❌ 局部变量，参数"漏"了
```
**后果**：`tudui.parameters()` 找不到它 → 优化器不更新 → 网络永远训不好，**且不报错**。
> **口诀：层一律 `self.xxx = 层(...)`。**

### 细节 3：网络可套网络
nn.Sequential 也是 Module，所以能塞进 `self.features = nn.Sequential(...)`。这就是网络能层层搭起来的原因。

## 六、常见报错对照
| 报错 | 原因 | 修法 |
|---|---|---|
| `cannot assign module before Module.__init__() call` | 漏 `super().__init__()` 或位置太后 | 提到第一行 |
| `'Tudui' object has no attribute 'forward'` | forward 名字拼错 | 必须叫 `forward` |
| `mat1 and mat2 shapes cannot be multiplied` | forward 里形状对不上 | 查每步 shape |
| 训不动、loss 不降 | 层没挂 `self.` | 全改 `self.xxx` |

## 七、FAQ
- **forward 里的 self 是什么？** 网络实例自己；`tudui(x)` 时自动传入，所以只传数据。
- **能写多个方法吗？** 能，`forward` 是必须的那个，其他自己 `self.方法名()` 调。
- **`__init__` 能不写层吗？** 能（+1 例子就没写），真实网络一定会写。
- **一定要继承 nn.Module 吗？** 不是必须，但你要自己管参数/搬 GPU/存模型。
- **两种调用输出一样，为什么较真？** `.train()/.eval()`、hook 都挂在 `__call__` 里，用 `.forward()` 会绕过，等用 Dropout/BatchNorm 时踩坑。

## 八、一句话总结
> **继承 nn.Module → `super().__init__()` 先跑 → 层挂 `self.` 上 → `forward` 描述数据流向 → 调用用 `网络(x)` 不用 `.forward(x)`。管参数、搬 GPU、存模型这些杂事它全包了。**

## 九、前瞻
下一步：`nn.Conv2d`（卷积层）、`nn.MaxPool2d`（池化层）、`nn.Linear`（全连接）等具体层，再学 `nn.Sequential` 把层串起来，然后接损失函数、优化器，跑通完整训练循环。
