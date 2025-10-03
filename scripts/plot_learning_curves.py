"""
Learning Curve Analysis Script
===============================
Visualize training and validation performance to detect overfitting/underfitting

This script:
1. Trains model while tracking train/val metrics
2. Plots learning curves for loss and accuracy
3. Analyzes overfitting/underfitting indicators
4. Provides diagnostic recommendations

Author: AD Detection Team
Date: October 2025
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
from imblearn.over_sampling import SMOTE

# Import from main module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from eeg_alzheimer_detection import (Config, HybridCNNLSTM, load_dataset, 
                                     device, compute_class_weights)

class LearningCurveTracker:
    """Track and visualize learning curves during training"""
    
    def __init__(self, config):
        self.config = config
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
            'epochs': []
        }
        
    def train_with_tracking(self, X, y, subjects):
        """Train model with full tracking of metrics"""
        print("="*80)
        print("LEARNING CURVE ANALYSIS - TRAINING WITH METRIC TRACKING")
        print("="*80)
        
        # Split data at subject level (80% train, 20% validation)
        unique_subjects = np.unique(subjects)
        subject_labels = np.array([y[subjects == subj][0] for subj in unique_subjects])
        
        train_subjects, val_subjects = train_test_split(
            unique_subjects, test_size=0.2, stratify=subject_labels, random_state=42
        )
        
        # Get samples for each split
        train_mask = np.isin(subjects, train_subjects)
        val_mask = np.isin(subjects, val_subjects)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]
        
        print(f"\nDataset Split:")
        print(f"Training:   {len(X_train)} samples (AD: {np.sum(y_train==1)}, HC: {np.sum(y_train==0)})")
        print(f"Validation: {len(X_val)} samples (AD: {np.sum(y_val==1)}, HC: {np.sum(y_val==0)})")
        
        # Apply SMOTE to training data only
        print("\nApplying SMOTE to training data...")
        X_train_flat = X_train.reshape(len(X_train), -1)
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train)
        X_train_balanced = X_train_balanced.reshape(-1, self.config.SEQUENCE_LENGTH, 
                                                     self.config.TOTAL_FEATURES)
        
        print(f"After SMOTE: {len(X_train_balanced)} samples (AD: {np.sum(y_train_balanced==1)}, HC: {np.sum(y_train_balanced==0)})")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_reshaped = X_train_balanced.reshape(-1, self.config.TOTAL_FEATURES)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(-1, self.config.SEQUENCE_LENGTH, 
                                                 self.config.TOTAL_FEATURES)
        
        X_val_reshaped = X_val.reshape(-1, self.config.TOTAL_FEATURES)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled.reshape(-1, self.config.SEQUENCE_LENGTH, 
                                            self.config.TOTAL_FEATURES)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train_balanced)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # Initialize model
        model = HybridCNNLSTM(self.config).to(device)
        
        # Loss and optimizer
        class_weights = compute_class_weights(y_train_balanced)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        
        # Training loop with tracking
        print(f"\nTraining for {self.config.NUM_EPOCHS} epochs...")
        print("="*80)
        
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, zero_division=0)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    
                    preds = torch.argmax(outputs, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())
            
            val_loss /= len(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            
            # Store metrics
            self.history['epochs'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:3d}/{self.config.NUM_EPOCHS}] | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        print("="*80)
        print("Training complete!")
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return model, self.history
    
    def plot_learning_curves(self, history):
        """Generate comprehensive learning curve visualizations"""
        print("\n" + "="*80)
        print("GENERATING LEARNING CURVE VISUALIZATIONS")
        print("="*80)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Loss curves
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_loss_curves(ax1, history)
        
        # 2. Accuracy curves
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_accuracy_curves(ax2, history)
        
        # 3. F1-Score curves
        ax3 = fig.add_subplot(gs[2, :2])
        self._plot_f1_curves(ax3, history)
        
        # 4. Overfitting indicator
        ax4 = fig.add_subplot(gs[0, 2])
        self._plot_overfitting_indicator(ax4, history)
        
        # 5. Final metrics comparison
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_final_metrics(ax5, history)
        
        # 6. Diagnostic text
        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_diagnostic_text(ax6, history)
        
        plt.suptitle('Learning Curve Analysis - Overfitting/Underfitting Detection', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        output_dir = self.config.OUTPUT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f'learning_curves_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Saved learning curves: {output_path}")
        plt.show()
        
        # Generate detailed analysis report
        self._generate_analysis_report(history, timestamp)
    
    def _plot_loss_curves(self, ax, history):
        """Plot training and validation loss"""
        epochs = history['epochs']
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        
        ax.plot(epochs, train_loss, 'o-', linewidth=2, markersize=6, 
               label='Training Loss', color='#2ecc71', alpha=0.8)
        ax.plot(epochs, val_loss, 's-', linewidth=2, markersize=6, 
               label='Validation Loss', color='#e74c3c', alpha=0.8)
        
        # Mark minimum validation loss
        min_val_idx = np.argmin(val_loss)
        ax.plot(epochs[min_val_idx], val_loss[min_val_idx], 'r*', 
               markersize=20, label=f'Best Val Loss (Epoch {epochs[min_val_idx]})')
        
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Loss', fontweight='bold', fontsize=12)
        ax.set_title('Loss Curves - Training vs Validation', fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add gap analysis
        final_gap = abs(train_loss[-1] - val_loss[-1])
        ax.text(0.02, 0.98, f'Final Gap: {final_gap:.4f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, fontweight='bold')
    
    def _plot_accuracy_curves(self, ax, history):
        """Plot training and validation accuracy"""
        epochs = history['epochs']
        train_acc = history['train_acc']
        val_acc = history['val_acc']
        
        ax.plot(epochs, train_acc, 'o-', linewidth=2, markersize=6, 
               label='Training Accuracy', color='#3498db', alpha=0.8)
        ax.plot(epochs, val_acc, 's-', linewidth=2, markersize=6, 
               label='Validation Accuracy', color='#9b59b6', alpha=0.8)
        
        # Mark maximum validation accuracy
        max_val_idx = np.argmax(val_acc)
        ax.plot(epochs[max_val_idx], val_acc[max_val_idx], 'g*', 
               markersize=20, label=f'Best Val Acc (Epoch {epochs[max_val_idx]})')
        
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
        ax.set_title('Accuracy Curves - Training vs Validation', fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add performance summary
        final_train = train_acc[-1]
        final_val = val_acc[-1]
        ax.text(0.02, 0.02, f'Final Train: {final_train:.4f}\nFinal Val: {final_val:.4f}', 
               transform=ax.transAxes, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
               fontsize=10, fontweight='bold')
    
    def _plot_f1_curves(self, ax, history):
        """Plot training and validation F1-scores"""
        epochs = history['epochs']
        train_f1 = history['train_f1']
        val_f1 = history['val_f1']
        
        ax.plot(epochs, train_f1, 'o-', linewidth=2, markersize=6, 
               label='Training F1-Score', color='#f39c12', alpha=0.8)
        ax.plot(epochs, val_f1, 's-', linewidth=2, markersize=6, 
               label='Validation F1-Score', color='#e67e22', alpha=0.8)
        
        # Mark maximum validation F1
        max_val_idx = np.argmax(val_f1)
        ax.plot(epochs[max_val_idx], val_f1[max_val_idx], 'b*', 
               markersize=20, label=f'Best Val F1 (Epoch {epochs[max_val_idx]})')
        
        ax.set_xlabel('Epoch', fontweight='bold', fontsize=12)
        ax.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
        ax.set_title('F1-Score Curves - Training vs Validation', fontweight='bold', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    def _plot_overfitting_indicator(self, ax, history):
        """Plot overfitting indicator metric"""
        epochs = history['epochs']
        
        # Calculate gap between train and val
        loss_gap = np.array(history['train_loss']) - np.array(history['val_loss'])
        acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        
        ax.plot(epochs, loss_gap, 'o-', linewidth=2, label='Loss Gap (Train-Val)', color='red')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Gap (Train - Val)', fontweight='bold')
        ax.set_title('Overfitting Indicator\n(Negative = Good)', fontweight='bold', fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Color zones
        ax.axhspan(-1, 0, alpha=0.1, color='green', label='Good')
        ax.axhspan(0, 1, alpha=0.1, color='red', label='Overfitting')
    
    def _plot_final_metrics(self, ax, history):
        """Plot final metrics comparison"""
        metrics = ['Accuracy', 'F1-Score']
        train_values = [history['train_acc'][-1], history['train_f1'][-1]]
        val_values = [history['val_acc'][-1], history['val_f1'][-1]]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_values, width, label='Training', 
                      alpha=0.8, color='#2ecc71')
        bars2 = ax.bar(x + width/2, val_values, width, label='Validation', 
                      alpha=0.8, color='#e74c3c')
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Final Metrics\nComparison', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(fontsize=9)
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    def _plot_diagnostic_text(self, ax, history):
        """Display diagnostic text analysis"""
        ax.axis('off')
        
        # Analyze the curves
        diagnosis = self._diagnose_fitting(history)
        
        # Create diagnostic text
        diagnostic_text = f"""
DIAGNOSTIC ANALYSIS
{'='*30}

Model Fit Status:
{diagnosis['status']}

Key Indicators:
• Final Train Loss: {history['train_loss'][-1]:.4f}
• Final Val Loss: {history['val_loss'][-1]:.4f}
• Loss Gap: {diagnosis['loss_gap']:.4f}

• Final Train Acc: {history['train_acc'][-1]:.4f}
• Final Val Acc: {history['val_acc'][-1]:.4f}
• Acc Gap: {diagnosis['acc_gap']:.4f}

Recommendation:
{diagnosis['recommendation']}

Confidence: {diagnosis['confidence']}
"""
        
        ax.text(0.05, 0.95, diagnostic_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor=diagnosis['color'], alpha=0.3))
    
    def _diagnose_fitting(self, history):
        """Diagnose overfitting, underfitting, or good fit"""
        loss_gap = history['train_loss'][-1] - history['val_loss'][-1]
        acc_gap = history['train_acc'][-1] - history['val_acc'][-1]
        
        train_loss_trend = history['train_loss'][-1] - history['train_loss'][0]
        val_loss_trend = history['val_loss'][-1] - history['val_loss'][0]
        
        diagnosis = {
            'loss_gap': loss_gap,
            'acc_gap': acc_gap
        }
        
        # Determine fit status
        if abs(loss_gap) < 0.1 and history['val_acc'][-1] > 0.7:
            diagnosis['status'] = "✅ GOOD FIT"
            diagnosis['color'] = 'lightgreen'
            diagnosis['confidence'] = "High"
            diagnosis['recommendation'] = "Model generalizes well.\nConsider deploying."
        
        elif loss_gap > 0.2 or acc_gap > 0.15:
            diagnosis['status'] = "⚠️ OVERFITTING"
            diagnosis['color'] = 'lightyellow'
            diagnosis['confidence'] = "Medium"
            diagnosis['recommendation'] = "Increase dropout,\nreduce complexity,\nor add more data."
        
        elif history['train_acc'][-1] < 0.7 and history['val_acc'][-1] < 0.7:
            diagnosis['status'] = "❌ UNDERFITTING"
            diagnosis['color'] = 'lightcoral'
            diagnosis['confidence'] = "Medium"
            diagnosis['recommendation'] = "Increase model capacity,\ntrain longer,\nor improve features."
        
        elif val_loss_trend > 0 and train_loss_trend < 0:
            diagnosis['status'] = "⚠️ OVERFITTING (Trend)"
            diagnosis['color'] = 'lightyellow'
            diagnosis['confidence'] = "High"
            diagnosis['recommendation'] = "Val loss increasing.\nUse early stopping\nor regularization."
        
        else:
            diagnosis['status'] = "✅ ACCEPTABLE FIT"
            diagnosis['color'] = 'lightblue'
            diagnosis['confidence'] = "Medium"
            diagnosis['recommendation'] = "Performance acceptable.\nMonitor on test set."
        
        return diagnosis
    
    def _generate_analysis_report(self, history, timestamp):
        """Generate detailed text report"""
        output_dir = self.config.OUTPUT_DIR
        report_path = os.path.join(output_dir, f'learning_curve_analysis_{timestamp}.txt')
        
        diagnosis = self._diagnose_fitting(history)
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LEARNING CURVE ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Epochs: {len(history['epochs'])}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("TRAINING PROGRESSION\n")
            f.write(f"{'='*80}\n")
            
            f.write(f"\nInitial Metrics (Epoch 1):\n")
            f.write(f"  Train Loss: {history['train_loss'][0]:.4f}\n")
            f.write(f"  Val Loss:   {history['val_loss'][0]:.4f}\n")
            f.write(f"  Train Acc:  {history['train_acc'][0]:.4f}\n")
            f.write(f"  Val Acc:    {history['val_acc'][0]:.4f}\n")
            
            f.write(f"\nFinal Metrics (Epoch {len(history['epochs'])}):\n")
            f.write(f"  Train Loss: {history['train_loss'][-1]:.4f}\n")
            f.write(f"  Val Loss:   {history['val_loss'][-1]:.4f}\n")
            f.write(f"  Train Acc:  {history['train_acc'][-1]:.4f}\n")
            f.write(f"  Val Acc:    {history['val_acc'][-1]:.4f}\n")
            f.write(f"  Train F1:   {history['train_f1'][-1]:.4f}\n")
            f.write(f"  Val F1:     {history['val_f1'][-1]:.4f}\n")
            
            f.write(f"\nBest Validation Metrics:\n")
            best_val_loss_idx = np.argmin(history['val_loss'])
            best_val_acc_idx = np.argmax(history['val_acc'])
            f.write(f"  Best Val Loss: {history['val_loss'][best_val_loss_idx]:.4f} at epoch {history['epochs'][best_val_loss_idx]}\n")
            f.write(f"  Best Val Acc:  {history['val_acc'][best_val_acc_idx]:.4f} at epoch {history['epochs'][best_val_acc_idx]}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("DIAGNOSTIC ANALYSIS\n")
            f.write(f"{'='*80}\n")
            
            f.write(f"\nFit Status: {diagnosis['status']}\n")
            f.write(f"Confidence: {diagnosis['confidence']}\n")
            
            f.write(f"\nKey Indicators:\n")
            f.write(f"  Loss Gap (Train - Val):     {diagnosis['loss_gap']:.4f}\n")
            f.write(f"  Accuracy Gap (Train - Val): {diagnosis['acc_gap']:.4f}\n")
            
            f.write(f"\nInterpretation:\n")
            if diagnosis['loss_gap'] < 0:
                f.write(f"  • Negative loss gap indicates validation loss < training loss\n")
                f.write(f"    This is unusual but can occur with dropout/regularization\n")
            elif diagnosis['loss_gap'] < 0.1:
                f.write(f"  • Small loss gap indicates good generalization\n")
            elif diagnosis['loss_gap'] < 0.2:
                f.write(f"  • Moderate loss gap - acceptable but monitor for overfitting\n")
            else:
                f.write(f"  • Large loss gap indicates overfitting\n")
            
            f.write(f"\nRecommendation:\n")
            f.write(f"  {diagnosis['recommendation']}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("EPOCH-BY-EPOCH METRICS\n")
            f.write(f"{'='*80}\n")
            f.write(f"\n{'Epoch':<8}{'Train Loss':<15}{'Val Loss':<15}{'Train Acc':<15}{'Val Acc':<15}\n")
            f.write("-"*80 + "\n")
            
            for i in range(len(history['epochs'])):
                f.write(f"{history['epochs'][i]:<8}"
                       f"{history['train_loss'][i]:<15.4f}"
                       f"{history['val_loss'][i]:<15.4f}"
                       f"{history['train_acc'][i]:<15.4f}"
                       f"{history['val_acc'][i]:<15.4f}\n")
        
        print(f"✅ Saved detailed report: {report_path}")


def main():
    """Main function"""
    print("="*80)
    print("LEARNING CURVE ANALYSIS")
    print("="*80)
    
    # Initialize configuration
    config = Config()
    
    # Load dataset
    print(f"\nLoading dataset from: {config.RAW_DATA_DIR}")
    X, y, subjects = load_dataset(config.RAW_DATA_DIR, config)
    
    if X is None or y is None or subjects is None:
        print("\n❌ Failed to load dataset.")
        return None
    
    print(f"✅ Dataset loaded successfully")
    print(f"   Total samples: {len(X)}")
    print(f"   AD samples: {np.sum(y == 1)}")
    print(f"   HC samples: {np.sum(y == 0)}")
    
    # Create tracker and train with metric tracking
    tracker = LearningCurveTracker(config)
    model, history = tracker.train_with_tracking(X, y, subjects)
    
    # Plot learning curves
    tracker.plot_learning_curves(history)
    
    print(f"\n{'='*80}")
    print("LEARNING CURVE ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Training completed with full metric tracking")
    print(f"✅ Learning curves generated and saved")
    print(f"✅ Diagnostic analysis completed")
    print(f"✅ Check results in: {config.OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    return history


if __name__ == "__main__":
    try:
        history = main()
        if history is None:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
