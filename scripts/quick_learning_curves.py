"""
Quick Learning Curve Analysis
==============================
Fast visualization of learning curves from existing training run

This script analyzes the model's training history to check for:
- Overfitting (high train accuracy, low validation accuracy)
- Underfitting (low train and validation accuracy)
- Good fit (similar train and validation performance)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Import from main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from eeg_alzheimer_detection import Config, HybridCNNLSTM, load_dataset, device

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def quick_train_and_plot(config, X, y, subjects, num_epochs=30):
    """Quick training with learning curve visualization"""
    
    print("\n" + "="*70)
    print("QUICK LEARNING CURVE ANALYSIS")
    print("="*70)
    
    # Simple train/val split
    unique_subjects = np.unique(subjects)
    subject_labels = np.array([y[subjects == subj][0] for subj in unique_subjects])
    
    train_subjects, val_subjects = train_test_split(
        unique_subjects, test_size=0.2, stratify=subject_labels, random_state=42
    )
    
    train_mask = np.isin(subjects, train_subjects)
    val_mask = np.isin(subjects, val_subjects)
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    print(f"\nDataset: Train={len(X_train)}, Val={len(X_val)}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, config.TOTAL_FEATURES)
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(-1, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)
    
    X_val_flat = X_val.reshape(-1, config.TOTAL_FEATURES)
    X_val_scaled = scaler.transform(X_val_flat).reshape(-1, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)
    
    # Tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Model
    model = HybridCNNLSTM(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Track metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'epochs': []
    }
    
    print(f"\nTraining for {num_epochs} epochs...")
    print("-"*70)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_labels.extend(y_batch.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Store
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | "
                  f"Loss: Train={train_loss:.4f} Val={val_loss:.4f} | "
                  f"Acc: Train={train_acc:.3f} Val={val_acc:.3f}")
    
    print("-"*70)
    return history


def plot_comprehensive_learning_curves(history, config):
    """Create comprehensive learning curve plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Learning Curve Analysis - Overfitting/Underfitting Detection', 
                fontsize=16, fontweight='bold')
    
    epochs = history['epochs']
    
    # 1. Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'o-', linewidth=2, 
            label='Training Loss', color='#2ecc71', markersize=5)
    ax1.plot(epochs, history['val_loss'], 's-', linewidth=2, 
            label='Validation Loss', color='#e74c3c', markersize=5)
    
    min_val_idx = np.argmin(history['val_loss'])
    ax1.plot(epochs[min_val_idx], history['val_loss'][min_val_idx], 'r*', 
            markersize=15, label=f'Best (Epoch {epochs[min_val_idx]})')
    
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Loss Curves', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add gap annotation
    final_gap = history['val_loss'][-1] - history['train_loss'][-1]
    ax1.text(0.02, 0.98, f'Final Gap: {final_gap:.4f}', 
            transform=ax1.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'o-', linewidth=2, 
            label='Training Accuracy', color='#3498db', markersize=5)
    ax2.plot(epochs, history['val_acc'], 's-', linewidth=2, 
            label='Validation Accuracy', color='#9b59b6', markersize=5)
    
    max_val_idx = np.argmax(history['val_acc'])
    ax2.plot(epochs[max_val_idx], history['val_acc'][max_val_idx], 'g*', 
            markersize=15, label=f'Best (Epoch {epochs[max_val_idx]})')
    
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Accuracy Curves', fontweight='bold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    # 3. Overfitting gap indicator
    ax3 = axes[1, 0]
    loss_gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    
    ax3.plot(epochs, loss_gap, 'o-', linewidth=2, color='red', markersize=5)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.fill_between(epochs, 0, loss_gap, where=(loss_gap > 0), 
                     alpha=0.3, color='red', label='Overfitting Zone')
    ax3.fill_between(epochs, loss_gap, 0, where=(loss_gap < 0), 
                     alpha=0.3, color='green', label='Good Zone')
    
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Val Loss - Train Loss', fontweight='bold')
    ax3.set_title('Overfitting Indicator', fontweight='bold', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Diagnostic summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Diagnose
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    acc_gap = final_train_acc - final_val_acc
    
    if acc_gap > 0.15:
        status = "⚠️ OVERFITTING DETECTED"
        color = 'lightyellow'
        advice = "• Train acc >> Val acc\n• Reduce model complexity\n• Add dropout/regularization\n• Get more training data"
    elif final_val_acc < 0.65 and final_train_acc < 0.65:
        status = "❌ UNDERFITTING DETECTED"
        color = 'lightcoral'
        advice = "• Low accuracy on both sets\n• Increase model capacity\n• Train for more epochs\n• Improve features"
    elif abs(acc_gap) < 0.1 and final_val_acc > 0.7:
        status = "✅ GOOD FIT!"
        color = 'lightgreen'
        advice = "• Model generalizes well\n• Similar train/val performance\n• Ready for deployment"
    else:
        status = "✅ ACCEPTABLE FIT"
        color = 'lightblue'
        advice = "• Reasonable performance\n• Monitor on test set\n• Consider fine-tuning"
    
    diagnostic = f"""
DIAGNOSTIC ANALYSIS
{'='*40}

Status: {status}

Final Metrics:
  Training Accuracy:    {final_train_acc:.4f}
  Validation Accuracy:  {final_val_acc:.4f}
  Accuracy Gap:         {acc_gap:.4f}

  Training Loss:        {history['train_loss'][-1]:.4f}
  Validation Loss:      {history['val_loss'][-1]:.4f}
  Loss Gap:             {final_gap:.4f}

Recommendations:
{advice}

Best Validation Accuracy:
  {history['val_acc'][max_val_idx]:.4f} at Epoch {epochs[max_val_idx]}
"""
    
    ax4.text(0.05, 0.95, diagnostic, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.5))
    
    plt.tight_layout()
    
    # Save
    output_dir = config.OUTPUT_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f'learning_curves_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved: {output_path}")
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nStatus: {status}")
    print(f"\nFinal Metrics:")
    print(f"  Train Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"  Val Accuracy:   {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"  Accuracy Gap:   {acc_gap:.4f}")
    print(f"\n  Train Loss:     {history['train_loss'][-1]:.4f}")
    print(f"  Val Loss:       {history['val_loss'][-1]:.4f}")
    print(f"  Loss Gap:       {final_gap:.4f}")
    print("="*70 + "\n")


def main():
    """Main function"""
    config = Config()
    
    # Load data
    print("\nLoading dataset...")
    X, y, subjects = load_dataset(config.RAW_DATA_DIR, config)
    
    if X is None:
        print("❌ Failed to load dataset")
        return
    
    print(f"✅ Loaded {len(X)} samples")
    
    # Train with tracking (reduced epochs for speed)
    history = quick_train_and_plot(config, X, y, subjects, num_epochs=30)
    
    # Plot results
    plot_comprehensive_learning_curves(history, config)
    
    print("\n🎉 Learning curve analysis complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
