# AI Learning：PyTorch 基础实验与学习笔记

这是我学习 PyTorch 过程中积累的实验代码和中文笔记，主要参考小土堆的 PyTorch 入门课程，并结合 AI 编程助手进行概念梳理、代码解释、实验设计和排错。

目前的重点是理解数据如何经过网络、损失如何产生梯度，以及优化器如何更新参数，再逐步把这些环节连接成能够独立编写的训练流程。

## 当前内容

| 主题 | 文件 | 实验内容 |
|---|---|---|
| 数据读取 | `dataset.py` | 自定义 `Dataset`，按目录读取图片与类别标签 |
| 卷积与形状 | `conv_demo.py` | 手算卷积、stride / padding、多通道与形状追踪 |
| 模型定义 | `model_demo.py` | `nn.Module`、`nn.Sequential`、CNN 前向与参数量 |
| 损失与梯度 | `loss_backward_demo.py` | 观察前向计算、反向传播、参数更新各自改变什么 |
| SGD | `sgd_first_steps.py` | 用两个回归样本观察权重、偏置和梯度 |
| 优化器对比 | `optimizer_demo.py` | 在相同初始化下比较 SGD、Momentum、Adam、AdamW |
| AdamW | `adamw_practice.py` | 权重衰减实验，以及训练损失和验证损失的区别 |
| 学习率调度 | `lr_schedule_practice.py` | 比较固定学习率与 StepLR 的训练过程 |
| 保存与恢复 | `checkpoint_practice.py` | 保存模型、优化器、调度器；比较恢复训练与连续训练 |
| 可视化 | `test_tb/tb.py` | TensorBoard 图像与标量记录 |
| 学习笔记 | [`笔记/`](笔记/) | transforms、Module、卷积、池化、线性层、初始化与求导 |

## 快速运行

建议使用已配置好 PyTorch 的 Python 环境，在仓库根目录运行。CPU 即可完成这里的基础实验。

```bash
python -m pip install -r requirements.txt
python model_demo.py
python loss_backward_demo.py
python optimizer_demo.py
python checkpoint_practice.py
```

`model_demo.py` 使用随机张量检查网络结构，预期输出形状为 `[4, 10]`。它尚未训练分类模型。

`checkpoint_practice.py` 先训练 4 轮，再保存状态并恢复到第 5、6 轮；最后与连续训练 6 轮的结果比较。通过时会输出：

```text
Check passed: resumed and uninterrupted training match.
```

这个对照是确定性的双样本回归练习。真实数据加载、随机增强、随机数状态和训练中途恢复，还需要在后续完整项目中继续验证。

### 最近一次运行检查

2026 年 10 月 14 日，在 Python 3.10.20、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128 环境中运行了上表的 10 个 Python 示例，均正常退出。Dataset 示例使用本地图片，TensorBoard 示例使用生成图片；检查点示例中的连续训练与恢复训练参数对照通过。该检查用于确认示例可运行，尚不代表完成了分类模型训练或性能评测。

## 图片数据与 TensorBoard

仓库不包含图片数据集。`dataset.py` 默认读取下面的本地结构，也可以通过 `--root` 指定训练目录：

```text
dataset_/train/
├── ants_image/
└── bees_image/
```

```bash
python dataset.py --root dataset_/train
```

这个练习返回 PIL 图片与字符串标签；后续分类训练还需加入 transforms、整数类别编号和 DataLoader。

TensorBoard 示例默认生成一张示意图片，无需下载数据；也可以传入自己的图片：

```bash
python test_tb/tb.py
python test_tb/tb.py --image path/to/image.jpg
tensorboard --logdir test_tb/logs
```

## 学习进度与下一步

目前已围绕网络层、损失、反向传播、优化器、验证指标和检查点进行基础学习与最小实验。正在将这些内容整合为自己编写的小型图像分类项目，继续练习数据处理、训练、验证、保存、推理和排错。

后续希望学习文本数据、Embedding、Attention 和 Transformer，再逐步实现小型语言模型 Oasis。相关内容属于后续计划，当前仓库尚未包含 Transformer、语言模型训练或 Agent 框架的实现。

## 学习与协作方式

我使用 Codex / Claude Code 辅助读写代码、解释报错、整理笔记和设计练习，再通过阅读、批注、修改和运行实验加深理解。仓库中包含学习示例和 AI 辅助编写的代码，记录的是学习过程与阶段实践。

Agent 的实际使用经验目前主要体现在编程辅助与项目工作流；工具调用、RAG 和自建 Agent 系统仍是后续学习内容。

笔记保留了学习时的解释和简化，其中的结论会随着进一步学习持续修订。
