#%% Preprocess and download datasets

# Variables
MODEL = 'ViT'

DATASET_PATH = './data/'
RESULT_PATH = './result_{}/'.format(MODEL)

# 创建结果目录
import os
os.makedirs(RESULT_PATH, exist_ok=True)

EPOCHS = 15
BATCH_SIZE = 64

LR = 1e-4  # 降低学习率
MOMENTUM = 0.9  # 保持动量不变

# 添加学习率预热
WARMUP_EPOCHS = 2
warmup_factor = lambda epoch: min(1.0, (epoch + 1) / WARMUP_EPOCHS)

# Import the necessary libraries
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import time
from tqdm import tqdm

# Loading the MNIST dataset
from torchvision import datasets, transforms

# Define a transform to normalize the data
if MODEL in ['LeNet', 'ViT']:  # ViT也使用原始28x28大小
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5))
    ])
else:  # AlexNet和ResNet使用224x224
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5), (0.5))
    ])

# Download and load the training data
trainset = datasets.MNIST(DATASET_PATH, download = True, train = True, transform = transform)
testset = datasets.MNIST(DATASET_PATH, download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)

#%% Define the model and the optimizer
from alexnet import AlexNet
from lenet import LeNet
from resnet import resnet18
from optim import SGD
from vit import ViT

if MODEL == 'LeNet':
    LR = 1e-6
    model = LeNet(input_channel=1, output_class=10)
elif MODEL == 'AlexNet':
    model = AlexNet(input_channel=1, output_class=10)
elif MODEL == 'ResNet18':
    LR = 5e-5
    model = resnet18(input_channel=1, output_class=10)
elif MODEL == 'ViT':
    # 删除这里的学习率设置，使用文件开头设置的值
    model = ViT(img_size=28, patch_size=7, in_channels=1, 
               embed_dim=64, num_heads=8, depth=6, 
               mlp_ratio=4., num_classes=10)
else:
    print('No such model {}, please indicate another'.format(MODEL))
optimizer = SGD(model, lr=LR, momentum=MOMENTUM)

#%% Training and evaluating process

# Initiate the timer to instrument the performance
timer_start = time.process_time_ns()
epoch_times = [timer_start]

train_losses, test_losses, accuracies = [], [], []

for e in range(EPOCHS):
    running_loss = 0
    print("Epoch: {:03d}/{:03d}..".format(e+1, EPOCHS))

    # 更新学习率
    current_lr = LR * warmup_factor(e)
    optimizer.lr = current_lr
    print(f"Current learning rate: {current_lr:.6f}")

    # Training pass
    print("Training pass:")
    for data in (tbar := tqdm(trainloader, total=len(trainloader))):
        images, labels = data[0].numpy(), data[1].numpy()

        prob, loss = model.forward(images, labels, train_mode=True)
        model.backward(loss)
        
        # 梯度裁剪
        for layer_grads in model.param_grads:
            if layer_grads:  # 如果有梯度
                for grad in layer_grads:
                    if grad is not None:
                        # 计算梯度范数
                        grad_norm = np.sqrt(np.sum(grad ** 2))
                        if grad_norm > 1.0:  # 如果梯度范数大于阈值
                            grad *= 1.0 / grad_norm  # 裁剪梯度
        
        optimizer.step()
        running_loss += np.mean(loss)  # 对loss取平均

        # 计算当前batch的准确率和平均损失
        batch_acc = np.mean((np.argmax(prob, axis=1) == labels).astype(float))
        batch_loss = np.mean(loss)
        tbar.set_description(f"Loss: {batch_loss:.3f}, Acc: {batch_acc:.3f}")
    
    # Testing pass
    print("Validation pass:")
    test_loss = 0
    accuracy = 0
    for data in tqdm(testloader, total=len(testloader)):
        images, labels = data[0].numpy(), data[1].numpy()
        
        prob, loss = model.forward(images, labels, train_mode=False)
        test_loss += np.mean(loss)  # 对loss取平均
        
        top_class = np.argmax(prob, axis=1)
        equals = (top_class == labels)
        accuracy += np.mean(equals.astype(float))

    train_losses.append(running_loss/len(trainloader))
    test_losses.append(test_loss/len(testloader))
    accuracies.append(accuracy/len(testloader))
    
    epoch_times.append(time.process_time_ns())
    print("Train loss: {:.3f}..".format(running_loss/len(trainloader)),
          "Test loss: {:.3f}..".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)),
          "Cur time(ns): {}".format(epoch_times[-1]))

#%% Evaluation

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(train_losses, label="Train loss")
ax.plot(test_losses, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross Entropy Loss")
ax.legend()
ax2 = ax.twinx()
ax2.plot(np.array(accuracies)*100, label="Accuracy", color='g')
ax2.set_ylabel("Accuracy (%)")
plt.title("Training procedure")
plt.savefig(RESULT_PATH + 'training_proc.png', dpi = 100)

proc_results = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'epoch_times': epoch_times,
    'accuracies': accuracies,
}

# print(proc_results)
with open(RESULT_PATH + 'torch_results.json', 'w+') as f:
    json.dump(proc_results, f)
