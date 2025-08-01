# Vision Transformer (ViT) for MNIST Classification

## 项目概述

本项目实现了Vision Transformer (ViT) 模型用于MNIST手写数字识别任务。ViT是一种将Transformer架构应用于计算机视觉的创新方法，通过将图像分割成patch序列，然后使用标准的Transformer编码器进行处理。

### 主要特性

- 🚀 **完整的ViT实现**: 包含完整的Vision Transformer架构
- 📊 **MNIST数据集支持**: 专门针对手写数字识别优化
- 🎯 **高精度**: 在MNIST测试集上达到97%+的准确率
- 📈 **训练可视化**: 自动生成训练损失和准确率图表
- 💾 **结果保存**: 自动保存训练结果和模型参数

## 项目结构

```
Network/
├── README.md                           # 项目说明文档
├── run_mnist.py                       # 主入口文件
├── vit.py                             # Vision Transformer实现
├── pylayer.py                         # 基础神经网络层实现
├── Vision_Transformer_for_MNIST.ipynb # 原始Jupyter notebook
├── alexnet.py                         # AlexNet模型实现
├── lenet.py                           # LeNet模型实现
├── resnet.py                          # ResNet模型实现
├── sequential.py                       # 序列模型实现
├── optim.py                           # 优化器实现
└── checker.ipynb                      # 检查工具
```

## 环境要求

### Python版本
- Python 3.7+

### 依赖包
```bash
pip install torch torchvision
pip install einops
pip install matplotlib
pip install numpy
pip install tqdm
```

## 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd Network
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行训练
```bash
python run_mnist.py
```

## 使用方法

### 基本训练

直接运行主入口文件开始训练：

```bash
python run_mnist.py
```

### 自定义参数

你可以修改 `run_mnist.py` 中的参数来调整训练：

```python
# 训练参数
EPOCHS = 10                    # 训练轮数
BATCH_SIZE_TRAIN = 100         # 训练批次大小
BATCH_SIZE_TEST = 1000         # 测试批次大小
LR = 0.003                     # 学习率

# 模型参数
IMAGE_SIZE = 28                # 图像大小
PATCH_SIZE = 4                 # Patch大小
NUM_CLASSES = 10               # 类别数
CHANNELS = 1                   # 输入通道数
DIM = 64                       # 模型维度
DEPTH = 6                      # Transformer深度
HEADS = 4                      # 注意力头数
MLP_DIM = 128                  # MLP隐藏层维度
```

## 模型架构

### Vision Transformer (ViT)

ViT模型包含以下主要组件：

1. **Patch Embedding**: 将28×28的图像分割成4×4的patches
2. **Position Embedding**: 为每个patch添加位置编码
3. **Transformer Encoder**: 包含6层Transformer块
4. **Classification Head**: 最终的分类层

### 模型参数

- **输入**: 28×28×1 (MNIST灰度图像)
- **Patch大小**: 4×4
- **Patch数量**: 49 (7×7)
- **模型维度**: 64
- **Transformer深度**: 6层
- **注意力头数**: 4
- **MLP隐藏层**: 128维
- **总参数量**: ~500K

## 训练过程

### 数据加载
- 自动下载MNIST数据集
- 应用标准化预处理
- 训练集: 60,000样本
- 测试集: 10,000样本

### 训练配置
- **优化器**: Adam
- **学习率**: 0.003
- **损失函数**: Cross Entropy Loss
- **批次大小**: 训练100，测试1000

### 训练监控
训练过程中会显示：
- 每个epoch的训练进度
- 实时损失值
- 测试准确率
- 训练时间

## 结果输出

### 自动生成的文件

训练完成后，会在 `result_ViT/` 目录下生成：

1. **training_results.png**: 训练损失和准确率图表
2. **results.json**: 详细的训练结果数据

### 结果示例

```
============================================================
Vision Transformer (ViT) for MNIST Classification
============================================================
Model: ViT
Epochs: 10
Batch Size (Train/Test): 100/1000
Learning Rate: 0.003
Image Size: 28x28
Patch Size: 4x4
Model Dimensions: dim=64, depth=6, heads=4, mlp_dim=128
============================================================

Loading MNIST dataset...
Training samples: 60000
Test samples: 10000

Creating ViT model...
Total trainable parameters: 499,722

Starting training...
------------------------------------------------------------
Epoch: 1
[    0/60000 (  0%)]  Loss: 2.7484
[10000/60000 ( 17%)]  Loss: 0.2415
...
Average test loss: 0.1931  Accuracy: 9404/10000 (94.04%)

...

============================================================
Training completed!
Final Test Accuracy: 0.9728 (97.28%)
Final Test Loss: 0.0886
Results saved to: ./result_ViT/
============================================================
```

## 文件说明

### 核心文件

#### `run_mnist.py`
主入口文件，包含：
- 完整的训练流程
- 参数配置
- 结果可视化
- 数据保存

#### `vit.py`
Vision Transformer实现，包含：
- `ViT`: 主要的ViT模型类
- `Transformer`: Transformer编码器
- `Attention`: 多头自注意力机制
- `FeedForward`: 前馈神经网络
- 数据加载和训练函数

#### `pylayer.py`
基础神经网络层实现，包含：
- `Linear`: 全连接层
- `Conv2d`: 2D卷积层
- `BatchNorm1d/2d`: 批量归一化
- `ReLU`: 激活函数
- `MaxPool2d`: 最大池化
- `CrossEntropyLossWithSoftmax`: 损失函数
- `BasicBlock/BottleNeck`: ResNet块

## 性能指标

### MNIST测试集结果
- **准确率**: 97.28%
- **训练时间**: ~60分钟 (CPU)
- **模型大小**: ~500K参数

### 训练曲线
- 收敛稳定，无过拟合
- 第6个epoch后准确率稳定在97%以上
- 损失函数平滑下降

## 扩展功能

### 自定义数据集
可以修改数据加载部分来支持其他数据集：

```python
# 在vit.py中修改load_mnist_data函数
def load_custom_data(batch_size_train=100, batch_size_test=1000):
    # 实现自定义数据加载
    pass
```

### 模型调优
可以通过调整以下参数来优化性能：

```python
# 增加模型容量
DIM = 128        # 增加模型维度
DEPTH = 12       # 增加Transformer深度
HEADS = 8        # 增加注意力头数

# 调整训练参数
EPOCHS = 20      # 增加训练轮数
LR = 0.001       # 调整学习率
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减少批次大小
   - 减少模型维度

2. **训练速度慢**
   - 使用GPU加速
   - 减少Transformer深度

3. **准确率不收敛**
   - 调整学习率
   - 增加训练轮数

### 依赖问题
```bash
# 如果遇到einops导入错误
pip install einops --upgrade

# 如果遇到torch版本问题
pip install torch==1.9.0 torchvision==0.10.0
```

## 贡献指南

欢迎提交Issue和Pull Request来改进项目：

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 详见LICENSE文件

## 致谢

- 感谢原始ViT论文作者
- 感谢PyTorch团队提供的优秀框架
- 感谢MNIST数据集提供者

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至：[your-email@example.com]

---

**注意**: 本项目主要用于学习和研究目的。在生产环境中使用前，请确保充分测试和验证。 