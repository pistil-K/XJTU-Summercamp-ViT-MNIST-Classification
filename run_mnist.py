#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision Transformer (ViT) for MNIST Handwritten Digit Recognition
Entry point for training and evaluating ViT model on MNIST dataset
"""

import torch
import matplotlib.pyplot as plt
import json
import time
import numpy as np
from tqdm import tqdm
import os

# Import ViT modules
from vit import (
    load_mnist_data, 
    create_vit_model, 
    train_vit_model, 
    evaluate, 
    count_parameters
)

def main():
    """Main function to train and evaluate ViT model on MNIST"""
    
    # Configuration
    MODEL = 'ViT'
    DATASET_PATH = './data/'
    RESULT_PATH = './result_{}/'.format(MODEL)
    
    # Create result directory if it doesn't exist
    os.makedirs(RESULT_PATH, exist_ok=True)
    
    # Training parameters
    EPOCHS = 10
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_TEST = 1000
    LR = 0.003
    
    # Model parameters
    IMAGE_SIZE = 28
    PATCH_SIZE = 4
    NUM_CLASSES = 10
    CHANNELS = 1
    DIM = 64
    DEPTH = 6
    HEADS = 4
    MLP_DIM = 128
    
    print("=" * 60)
    print("Vision Transformer (ViT) for MNIST Classification")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size (Train/Test): {BATCH_SIZE_TRAIN}/{BATCH_SIZE_TEST}")
    print(f"Learning Rate: {LR}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Patch Size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Model Dimensions: dim={DIM}, depth={DEPTH}, heads={HEADS}, mlp_dim={MLP_DIM}")
    print("=" * 60)
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(
        batch_size_train=BATCH_SIZE_TRAIN, 
        batch_size_test=BATCH_SIZE_TEST
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create ViT model
    print("\nCreating ViT model...")
    model = create_vit_model(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=NUM_CLASSES,
        channels=CHANNELS,
        dim=DIM,
        depth=DEPTH,
        heads=HEADS,
        mlp_dim=MLP_DIM
    )
    
    # Count parameters
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Training and evaluation
    print("\nStarting training...")
    print("-" * 60)
    
    train_losses, test_losses, accuracies = train_vit_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=EPOCHS,
        lr=LR
    )
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_loss, final_accuracy = evaluate(model, test_loader, [])
    
    # Plot training results
    print("\nPlotting training results...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(test_losses, label='Test Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot([acc * 100 for acc in accuracies], label='Test Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(RESULT_PATH + 'training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'model': MODEL,
        'epochs': EPOCHS,
        'batch_size_train': BATCH_SIZE_TRAIN,
        'batch_size_test': BATCH_SIZE_TEST,
        'learning_rate': LR,
        'image_size': IMAGE_SIZE,
        'patch_size': PATCH_SIZE,
        'model_params': {
            'dim': DIM,
            'depth': DEPTH,
            'heads': HEADS,
            'mlp_dim': MLP_DIM
        },
        'total_parameters': total_params,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'accuracies': accuracies,
        'final_accuracy': final_accuracy,
        'final_loss': final_loss
    }
    
    with open(RESULT_PATH + 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Final Test Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    print(f"Final Test Loss: {final_loss:.4f}")
    print(f"Results saved to: {RESULT_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    main()
