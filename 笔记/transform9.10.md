# Transform 与 `__call__` 机制笔记（9.10）

## 一、transforms 是什么
- `torchvision.transforms` 是一堆**图片预处理工具**（都是**类**，不是函数）
- 作用：把原始图片（PIL 图 / numpy 数组）加工成**模型能吃的、标准化的 tensor**
- 流程：`原始图片 → transforms 处理 → tensor → 喂给网络`

## 二、万能套路（所有 transform 通用）
```python
工具 = transforms.某某Transform(参数)   # 造实例
结果 = 工具(图片)                       # 调用实例去处理
```
> 不要写成 `transforms.ToTensor(img)`，要先用 `()` 造实例，再传图。

## 三、常用 transforms 速查表
| Transform | 作用 | 输入 → 输出 | 记忆点 |
|---|---|---|---|
| `ToTensor()` | 转 tensor | PIL/numpy → tensor | 像素 [0,255]→[0,1]，通道 HWC→CHW |
| `Normalize(mean, std)` | 标准化 | 必须 tensor | 公式 `(x-mean)/std`，按通道给 |
| `Resize((h,w))` | 改尺寸 | PIL → PIL | 是 **(高,宽)**，不是(宽,高) |
| `RandomCrop((h,w))` | 随机裁剪 | PIL → PIL | 裁到 (h,w) |
| `Compose([...])` | 按顺序组合 | 看首尾 | 上一个的输出 = 下一个的输入 |

## 四、Compose 与顺序（最易错）
```python
trans = transforms.Compose([
    transforms.Resize((224,224)),                          # 几何变换在前（吃 PIL）
    transforms.ToTensor(),                                 # 转 tensor 居中
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),     # 标准化收尾（吃 tensor）
])
```
**口诀**：几何变换在前 → ToTensor 居中 → Normalize 收尾。

## 五、完整例子
```python
from PIL import Image
from torchvision import transforms

img = Image.open("test.jpg")
trans = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
])
img_tensor = trans(img)   # shape: [3,224,224]
```

## 六、核心 Python 概念区分

| 概念 | 一句话 | 何时运行 |
|---|---|---|
| 类 class | 图纸/模板 | 不运行 |
| 实例 instance | 按图纸造出的实物 | `类()` 造出来的那刻 |
| `__init__` | 初始化：造实例时自动执行 | 写 `类()` 自动触发 |
| 属性 attribute | 存在实例身上的数据（`self.xxx`） | 存着 |
| 方法 method | 类里的函数（首参 `self`） | `实例.方法()` 主动调 |
| `__call__` | 让实例能像函数被调用 | 写 `实例()` 触发 |

**完整示例：**
```python
class Person:
    def __init__(self, name, age):   # __init__ 造实例时自动跑
        self.name = name             # 属性
        self.age = age               # 属性
    def say_hi(self):                # 方法
        print(f"我是{self.name}，{self.age}岁")
    def __call__(self):              # 让实例能被调用
        print(f"{self.name} 被调用了")

p = Person("小明", 18)   # 触发 __init__
p.say_hi()               # 调用方法
p()                      # 触发 __call__
```

**关键分工**：`__init__` 负责**存配置**，`__call__` 负责**干正事**。
（还原 Normalize：`__init__(mean,std)` 存进 `self.mean/self.std`，`__call__(tensor)` 用它们算 `(tensor-mean)/std`）

## 七、`__call__` 常见疑问

### Q1：`__call__` 能多次使用吗？
- **定义**只能一次（写两个，后一个覆盖前一个）
- **调用**可以无数次，每次跑同一段代码，**靠参数区分**：
```python
class Adder:
    def __init__(self, base): self.base = base
    def __call__(self, x): return self.base + x
a = Adder(10)
a(5)    # 15
a(100)  # 110
```
- 想做不同的事 → 写多个**不同名字的普通方法**，靠方法名区分。

### Q2：为什么用 `__call__` 而不是方法？
**它能被方法替代，功能一样。** 用它有 3 个理由：
1. **统一套路**：所有工具都是 `工具(图)`，不用记各自方法名
2. **配置分离**：造实例时定死参数，之后只传数据（"带配置的函数"）
3. **方便组合**：`Compose` 能统一 `t(img)` 循环处理

### Q3：方法名不同为什么循环不优雅？
- 方法名各不同 → 循环体 `t.??? (img)` 写不出，得 if/else 一个个查，每加工具多写判断
- 统一方法名（都叫 `apply`）→ 可行，但"统一叫什么"还得约定记忆
- `__call__` → 用"调用"这个天然规则，连名字都不用起

## 八、一句话总结
> transforms = 造工具(`__init__` 存配置) → 用工具(`__call__` 干正事)；`Compose` 按顺序排队，几何变换在前、ToTensor 居中、Normalize 收尾。

## 九、前瞻（复现项目时会遇到）
复现预训练模型（ResNet 等）时，`Normalize` 的 mean/std 要用 **ImageNet 固定值**（如 `(0.485,0.456,0.406)`），必须照抄。
