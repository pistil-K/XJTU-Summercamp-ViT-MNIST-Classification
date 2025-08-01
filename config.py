#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT) 配置文件
集中管理项目的所有参数和配置
"""

import os

class Config:
    """项目配置类"""
    
    # 项目基本信息
    PROJECT_NAME = "Vision Transformer for MNIST"
    VERSION = "1.0.0"
    AUTHOR = "Your Name"
    
    # 路径配置
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULT_DIR = os.path.join(BASE_DIR, "result_ViT")
    
    # 数据配置
    DATASET_NAME = "MNIST"
    IMAGE_SIZE = 28
    NUM_CLASSES = 10
    CHANNELS = 1
    
    # 训练配置
    EPOCHS = 10
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_TEST = 1000
    LEARNING_RATE = 0.003
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # 模型配置
    PATCH_SIZE = 4
    DIM = 64
    DEPTH = 6
    HEADS = 4
    MLP_DIM = 128
    DROPOUT = 0.0
    EMB_DROPOUT = 0.0
    
    # 优化器配置
    OPTIMIZER = "Adam"
    SCHEDULER = None
    SCHEDULER_STEP_SIZE = 30
    SCHEDULER_GAMMA = 0.1
    
    # 数据增强配置
    USE_DATA_AUGMENTATION = False
    RANDOM_ROTATION = 10
    RANDOM_TRANSLATION = 0.1
    RANDOM_SCALE = 0.1
    
    # 日志配置
    LOG_LEVEL = "INFO"
    SAVE_MODEL = True
    SAVE_BEST_MODEL = True
    SAVE_LAST_MODEL = True
    
    # 可视化配置
    PLOT_TRAINING_CURVE = True
    PLOT_CONFUSION_MATRIX = True
    SAVE_PLOTS = True
    
    # 硬件配置
    DEVICE = "cpu"  # "cpu" or "cuda"
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # 随机种子
    RANDOM_SEED = 42
    
    @classmethod
    def get_model_config(cls):
        """获取模型配置"""
        return {
            "image_size": cls.IMAGE_SIZE,
            "patch_size": cls.PATCH_SIZE,
            "num_classes": cls.NUM_CLASSES,
            "channels": cls.CHANNELS,
            "dim": cls.DIM,
            "depth": cls.DEPTH,
            "heads": cls.HEADS,
            "mlp_dim": cls.MLP_DIM,
            "dropout": cls.DROPOUT,
            "emb_dropout": cls.EMB_DROPOUT
        }
    
    @classmethod
    def get_training_config(cls):
        """获取训练配置"""
        return {
            "epochs": cls.EPOCHS,
            "batch_size_train": cls.BATCH_SIZE_TRAIN,
            "batch_size_test": cls.BATCH_SIZE_TEST,
            "learning_rate": cls.LEARNING_RATE,
            "momentum": cls.MOMENTUM,
            "weight_decay": cls.WEIGHT_DECAY,
            "optimizer": cls.OPTIMIZER,
            "scheduler": cls.SCHEDULER,
            "scheduler_step_size": cls.SCHEDULER_STEP_SIZE,
            "scheduler_gamma": cls.SCHEDULER_GAMMA
        }
    
    @classmethod
    def get_data_config(cls):
        """获取数据配置"""
        return {
            "dataset_name": cls.DATASET_NAME,
            "image_size": cls.IMAGE_SIZE,
            "num_classes": cls.NUM_CLASSES,
            "channels": cls.CHANNELS,
            "use_data_augmentation": cls.USE_DATA_AUGMENTATION,
            "random_rotation": cls.RANDOM_ROTATION,
            "random_translation": cls.RANDOM_TRANSLATION,
            "random_scale": cls.RANDOM_SCALE
        }
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("=" * 60)
        print(f"项目: {cls.PROJECT_NAME}")
        print(f"版本: {cls.VERSION}")
        print(f"作者: {cls.AUTHOR}")
        print("=" * 60)
        
        print("\n模型配置:")
        model_config = cls.get_model_config()
        for key, value in model_config.items():
            print(f"  {key}: {value}")
        
        print("\n训练配置:")
        training_config = cls.get_training_config()
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        
        print("\n数据配置:")
        data_config = cls.get_data_config()
        for key, value in data_config.items():
            print(f"  {key}: {value}")
        
        print("\n其他配置:")
        print(f"  设备: {cls.DEVICE}")
        print(f"  随机种子: {cls.RANDOM_SEED}")
        print(f"  数据目录: {cls.DATA_DIR}")
        print(f"  结果目录: {cls.RESULT_DIR}")
        print("=" * 60)

# 创建默认配置实例
config = Config()

if __name__ == "__main__":
    config.print_config() 