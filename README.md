# XJTU-Summercamp-ViT-MNIST-Classification

## 项目概述
这是一个基于NumPy实现的深度学习框架，专门用于MNIST手写数字分类任务。项目实现了从基础层到复杂网络架构的完整深度学习系统，通过手写实现各种网络层，帮助理解深度学习的核心概念和实现细节。

## 目录结构
```
XJTU-Summercamp-ViT-MNIST-Classification/
├── README.md                    # 项目说明文件
└── NetworkNumpy/
    └── Network/
        ├── run_mnist.py         # 主运行脚本
        ├── pylayer.py           # 核心网络层实现
        ├── sequential.py        # 序列模型容器
        ├── optim.py            # 优化器实现
        ├── lenet.py            # LeNet网络架构
        ├── alexnet.py          # AlexNet网络架构
        ├── resnet.py           # ResNet网络架构
        ├── vit.py              # Vision Transformer实现
        └── checker.ipynb       # 测试验证脚本
```

## Vision Transformer实现详解

### 1. 网络架构和数据流
```
输入图像 (B=64, C=1, H=28, W=28)
│
├─> PatchEmbedding
│   └─> 分割patches: (64, 1, 28, 28) -> (64, 16, 49)  # 16个7x7的patches
│   └─> 线性投影: (64, 16, 49) -> (64, 16, 64)  # 投影到embed_dim维度
│
├─> Transformer Encoder (x6)
│   ├─> Multi-Head Attention
│   │   ├─> Q,K,V投影: (64, 16, 64) -> 3x(64, 16, 64)
│   │   ├─> 重塑多头: (64, 16, 64) -> (64, 8, 16, 8)  # 8个注意力头
│   │   ├─> 注意力计算: (64, 8, 16, 16)
│   │   └─> 输出投影: (64, 16, 64)
│   │
│   ├─> BatchNorm1d
│   │   └─> (64, 16, 64) -> (64, 16, 64)
│   │
│   └─> MLP
│       ├─> FC1: (64*16, 64) -> (64*16, 256)
│       ├─> ReLU
│       └─> FC2: (64*16, 256) -> (64*16, 64)
│
├─> Global Average Pooling
│   └─> (64, 16, 64) -> (64, 64)
│
└─> 分类头
    └─> Linear: (64, 64) -> (64, 10)
```

### 2. 实现过程中的问题和解决方案

#### 问题1: 循环导入
- **问题描述**: `vit.py` 和 `pylayer.py` 之间存在循环导入
- **原因**: 模块间相互依赖导致导入死锁
- **解决方案**: 
  ```python
  # vit.py
  import pylayer as L  # 直接导入基础层
  from sequential import Sequential  # 导入Sequential类
  ```

#### 问题2: 维度不匹配
- **问题描述**: `ValueError: shapes (1024,4096) and (64,64) not aligned`
- **原因**: PatchEmbedding输出维度计算错误
- **解决方案**: 
  ```python
  class PatchEmbedding:
      def forward(self, x):
          B, C, H, W = x.shape  # (64, 1, 28, 28)
          x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
          x = x.transpose(0, 2, 4, 3, 5, 1)  # 调整维度顺序
          x = x.reshape(B, self.n_patches, -1)  # (64, 16, 49)
  ```

#### 问题3: 梯度结构不匹配（元组问题）
- **问题描述**: `AttributeError: 'tuple' object has no attribute 'shape'`
- **原因**: backward返回的梯度结构与params_ref不匹配
- **解决方案**: 
  ```python
  class TransformerBlock:
      def backward(self, grad_output):
          # 返回扁平元组而不是嵌套结构
          return grad_output, (grad_w1, grad_b1, grad_w2, grad_b2)
  ```

#### 问题4: 梯度收集问题
- **问题描述**: `AttributeError: 'list' object has no attribute 'shape'`
- **原因**: Sequential.backward中梯度收集方式不正确
- **解决方案**: 
  ```python
  class Sequential:
      def backward(self, grad):
          if len(bwd_ret) > 1:
              grad_params = list(bwd_ret[1])  # 将元组转换为列表
              grads.append(grad_params)
  ```

### 3. 关键数据结构

#### 参数引用结构 (params_ref)
```python
[
    [weight1, bias1],     # 第一层参数
    [weight2, bias2],     # 第二层参数
    ...
]
```

#### 梯度结构 (param_grads)
```python
[
    [grad_w1, grad_b1],   # 第一层梯度
    [grad_w2, grad_b2],   # 第二层梯度
    ...
]
```

### 4. 最佳实践和经验总结

1. **维度处理**:
   - 始终清晰记录每一层的输入输出维度
   - 在关键位置添加维度检查和断言
   - 注意batch维度的处理

2. **梯度流**:
   - 保持梯度结构与参数结构一致
   - 注意梯度的累积和传播
   - 正确处理残差连接

3. **模块化设计**:
   - 避免循环依赖
   - 清晰的接口定义
   - 一致的数据结构

4. **调试技巧**:
   - 使用print语句跟踪维度变化
   - 添加shape断言检查
   - 分段验证梯度计算

## 核心组件详细分析

### 1. 主运行脚本 (`run_mnist.py`)
- **功能**: 整个项目的入口点，负责数据加载、模型训练和评估
- **主要特性**:
  - 支持多种模型选择（LeNet、AlexNet、ResNet18）
  - 自动下载和预处理MNIST数据集
  - 完整的训练和验证循环
  - 性能监控和结果可视化
  - 结果保存功能

### 2. 核心网络层实现 (`pylayer.py`)
这是项目的核心文件，包含了所有基础网络层的实现：

**基础层**:
- `Linear`: 全连接层，实现 `y = xW + b`
- `ReLU`: 激活函数，实现 `y = max(x, 0)`
- `CrossEntropyLossWithSoftmax`: 损失函数，结合softmax和交叉熵

**卷积相关层**:
- `Conv2d`: 2D卷积层，支持padding、stride等参数
- `MaxPool2d`: 最大池化层
- `Flatten`: 展平层，用于连接卷积层和全连接层

**批归一化层**:
- `BatchNorm1d`: 1D批归一化
- `BatchNorm2d`: 2D批归一化，继承自BatchNorm1d

**残差块**:
- `BasicBlock`: ResNet的基础残差块
- `BottleNeck`: ResNet的瓶颈残差块

**辅助函数**:
- `im2col`/`col2im`: 用于卷积操作的矩阵变换

### 3. 序列模型容器 (`sequential.py`)
- **功能**: 提供统一的模型接口，管理多个层的顺序执行
- **特性**:
  - 自动处理前向传播和反向传播
  - 支持训练和测试模式切换
  - 统一的参数梯度管理

### 4. 优化器 (`optim.py`)
- **SGD类**: 实现带动量的随机梯度下降
- **特性**:
  - 支持动量参数
  - 自动管理所有可训练参数
  - 支持多种层类型的参数更新

### 5. 网络架构实现

**LeNet** (`lenet.py`):
- 经典的LeNet-5架构
- 适用于28×28的MNIST图像
- 结构：Conv2d → ReLU → MaxPool2d → Conv2d → ReLU → MaxPool2d → Flatten → Linear → ReLU → Linear → ReLU → Linear

**AlexNet** (`alexnet.py`):
- 经典的AlexNet架构
- 适用于224×224的图像
- 包含多个卷积层、池化层和全连接层

**ResNet** (`resnet.py`):
- 实现了完整的ResNet系列
- 支持ResNet18、34、50、101、152
- 包含BasicBlock和BottleNeck两种残差块
- 支持残差连接和批归一化

### 6. 测试验证 (`checker.ipynb`)
- Jupyter notebook格式的测试脚本
- 用于验证实现的正确性
- 与PyTorch实现进行对比测试

## 技术特点

1. **纯NumPy实现**: 所有计算都基于NumPy，不依赖深度学习框架
2. **模块化设计**: 每个层都是独立的类，便于扩展和维护
3. **完整的反向传播**: 实现了所有层的梯度计算
4. **多种网络架构**: 支持从简单到复杂的多种网络结构
5. **性能监控**: 包含训练时间、损失、准确率等指标的监控

## 使用方式
项目通过修改`run_mnist.py`中的`MODEL`变量来选择不同的网络架构，支持：
- `'LeNet'`: 轻量级网络，适合快速实验
- `'AlexNet'`: 中等复杂度网络
- `'ResNet18'`: 深度残差网络，性能较好

## 运行环境要求
- Python 3.x
- NumPy
- PyTorch (用于数据加载)
- Matplotlib (用于结果可视化)
- tqdm (用于进度条显示)

## 快速开始
1. 确保安装了所需的依赖包
2. 修改`run_mnist.py`中的`MODEL`变量选择网络架构
3. 运行`python run_mnist.py`开始训练
4. 查看`result_{MODEL}/`目录下的训练结果和可视化图表
