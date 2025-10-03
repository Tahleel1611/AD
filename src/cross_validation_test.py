"""
Cross-Validation Testing Script
================================
Independent testing of the trained model using cross-validation

This script performs:
1. Loads the trained model
2. Performs cross-validation on test data
3. Generates performance metrics and visualizations
4. Validates model generalization

Author: AD Detection Team
Date: October 2024
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

# Import from main module
from eeg_alzheimer_detection import (Config, HybridCNNLSTM, load_dataset, 
                                     device, evaluate)

class CrossValidationTester:
    """Cross-validation testing for trained model"""
    
    def __init__(self, model_path, config):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.results = []
        
    def load_trained_model(self):
        """Load the trained model"""
        print("="*60)
        print("LOADING TRAINED MODEL")
        print("="*60)
        
        if not os.path.exists(self.model_path):
            print(f"❌ Model file not found: {self.model_path}")
            return False
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=device)
            
            # Print model information
            print(f"✅ Model loaded from: {self.model_path}")
            if 'best_val_acc' in checkpoint:
                print(f"   Training accuracy: {checkpoint['best_val_acc']:.2f}%")
            if 'best_f1' in checkpoint:
                print(f"   Training F1-Score: {checkpoint['best_f1']:.4f}")
            
            # Create model with loaded configuration
            if 'config' in checkpoint:
                # Use saved config if available
                saved_config = checkpoint['config']
                print("\n✅ Using saved model configuration:")
                for key, value in saved_config.items():
                    print(f"   {key}: {value}")
            
            # Initialize model
            self.model = HybridCNNLSTM(self.config).to(device)
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            print("\n✅ Model loaded and ready for testing")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def perform_cross_validation_test(self, X, y, subjects):
        """Perform k-fold cross-validation testing"""
        print("\n" + "="*60)
        print("CROSS-VALIDATION TESTING")
        print("="*60)
        
        # Subject-level stratified K-fold
        unique_subjects = np.unique(subjects)
        subject_labels = np.array([y[subjects == subj][0] for subj in unique_subjects])
        
        skf = StratifiedKFold(n_splits=self.config.K_FOLDS, shuffle=True, random_state=42)
        
        print(f"\nPerforming {self.config.K_FOLDS}-fold cross-validation...")
        print(f"Total unique subjects: {len(unique_subjects)}")
        print(f"AD subjects: {np.sum(subject_labels == 1)}")
        print(f"HC subjects: {np.sum(subject_labels == 0)}")
        
        # Store results
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(unique_subjects, subject_labels)):
            print(f"\n{'-'*60}")
            print(f"FOLD {fold + 1}/{self.config.K_FOLDS}")
            print(f"{'-'*60}")
            
            # Get test subjects for this fold
            test_subjects = unique_subjects[test_idx]
            test_mask = np.isin(subjects, test_subjects)
            
            X_test_fold = X[test_mask]
            y_test_fold = y[test_mask]
            
            print(f"Test samples: {len(X_test_fold)} (AD: {np.sum(y_test_fold==1)}, HC: {np.sum(y_test_fold==0)})")
            
            # Standardize features
            scaler = StandardScaler()
            X_test_reshaped = X_test_fold.reshape(-1, self.config.TOTAL_FEATURES)
            X_test_scaled = scaler.fit_transform(X_test_reshaped)
            X_test_scaled = X_test_scaled.reshape(-1, self.config.SEQUENCE_LENGTH, self.config.TOTAL_FEATURES)
            
            # Convert to tensors
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.LongTensor(y_test_fold)
            
            # Create data loader
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
            test_loader = DataLoader(test_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
            
            # Evaluate on this fold
            y_true, y_pred, y_prob = evaluate(self.model, test_loader)
            
            # Store results
            all_predictions.extend(y_pred)
            all_true_labels.extend(y_true)
            all_probabilities.extend(y_prob)
            
            # Calculate metrics for this fold
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.0
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Store fold results
            self.results.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            })
            
            print(f"\nFold {fold+1} Results:")
            print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
            print(f"  F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
            print(f"  AUC:       {auc:.4f} ({auc*100:.2f}%)")
            print(f"\nConfusion Matrix:")
            print(f"  TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
            print(f"  FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")
        
        # Calculate overall metrics
        self.calculate_overall_metrics(all_true_labels, all_predictions, all_probabilities)
        
        return self.results
    
    def calculate_overall_metrics(self, y_true, y_pred, y_prob):
        """Calculate overall performance metrics"""
        print(f"\n{'='*60}")
        print("OVERALL CROSS-VALIDATION RESULTS")
        print(f"{'='*60}")
        
        # Overall metrics
        overall_accuracy = accuracy_score(y_true, y_pred)
        overall_precision = precision_score(y_true, y_pred, zero_division=0)
        overall_recall = recall_score(y_true, y_pred, zero_division=0)
        overall_f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            overall_auc = roc_auc_score(y_true, y_prob)
        except:
            overall_auc = 0.0
        
        overall_cm = confusion_matrix(y_true, y_pred)
        
        print(f"\nOverall Performance Metrics:")
        print(f"  Accuracy:  {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
        print(f"  Precision: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
        print(f"  Recall:    {overall_recall:.4f} ({overall_recall*100:.2f}%)")
        print(f"  F1-Score:  {overall_f1:.4f} ({overall_f1*100:.2f}%)")
        print(f"  AUC:       {overall_auc:.4f} ({overall_auc*100:.2f}%)")
        
        print(f"\nOverall Confusion Matrix:")
        print(f"  TN: {overall_cm[0,0]:3d}  FP: {overall_cm[0,1]:3d}")
        print(f"  FN: {overall_cm[1,0]:3d}  TP: {overall_cm[1,1]:3d}")
        
        # Per-fold statistics
        print(f"\nPer-Fold Statistics:")
        avg_accuracy = np.mean([r['accuracy'] for r in self.results])
        std_accuracy = np.std([r['accuracy'] for r in self.results])
        
        avg_precision = np.mean([r['precision'] for r in self.results])
        std_precision = np.std([r['precision'] for r in self.results])
        
        avg_recall = np.mean([r['recall'] for r in self.results])
        std_recall = np.std([r['recall'] for r in self.results])
        
        avg_f1 = np.mean([r['f1'] for r in self.results])
        std_f1 = np.std([r['f1'] for r in self.results])
        
        avg_auc = np.mean([r['auc'] for r in self.results])
        std_auc = np.std([r['auc'] for r in self.results])
        
        print(f"  Accuracy:  {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"  Precision: {avg_precision:.4f} ± {std_precision:.4f}")
        print(f"  Recall:    {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"  F1-Score:  {avg_f1:.4f} ± {std_f1:.4f}")
        print(f"  AUC:       {avg_auc:.4f} ± {std_auc:.4f}")
        
        # Store overall results
        self.overall_results = {
            'overall_accuracy': overall_accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'overall_auc': overall_auc,
            'overall_cm': overall_cm,
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print(f"\n{'='*60}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*60}")
        
        output_dir = self.config.OUTPUT_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_confusion_matrix(ax1)
        
        # 2. ROC Curve
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_roc_curve(ax2)
        
        # 3. Per-fold metrics
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_fold_metrics(ax3)
        
        # 4. Metric comparison bars
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_metric_comparison(ax4)
        
        # 5. Precision-Recall curve
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_precision_recall_curve(ax5)
        
        # 6. Per-fold confusion matrices
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_fold_confusion_summary(ax6)
        
        # 7. Performance over folds
        ax7 = fig.add_subplot(gs[2, :])
        self._plot_performance_over_folds(ax7)
        
        plt.suptitle('Cross-Validation Test Results - Comprehensive Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        # Save figure
        output_path = os.path.join(output_dir, f'cv_test_plot_{timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"✅ Saved visualization: {output_path}")
        
        # Generate classification report
        self._save_classification_report(timestamp)
    
    def _plot_confusion_matrix(self, ax):
        """Plot overall confusion matrix"""
        cm = self.overall_results['overall_cm']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=False, square=True, linewidths=2, linecolor='black')
        
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title('Overall Confusion Matrix', fontweight='bold')
        ax.set_xticklabels(['HC', 'AD'])
        ax.set_yticklabels(['HC', 'AD'])
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                       ha='center', va='center', fontsize=9, color='gray')
    
    def _plot_roc_curve(self, ax):
        """Plot ROC curve"""
        y_true = self.overall_results['y_true']
        y_prob = self.overall_results['y_prob']
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = self.overall_results['overall_auc']
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curve', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    def _plot_fold_metrics(self, ax):
        """Plot metrics for each fold"""
        folds = [r['fold'] for r in self.results]
        f1_scores = [r['f1'] for r in self.results]
        accuracies = [r['accuracy'] for r in self.results]
        
        ax.plot(folds, f1_scores, marker='o', linewidth=2, markersize=8, 
               label='F1-Score', color='green')
        ax.plot(folds, accuracies, marker='s', linewidth=2, markersize=8,
               label='Accuracy', color='blue')
        
        ax.set_xlabel('Fold', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Performance by Fold', fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(folds)
    
    def _plot_metric_comparison(self, ax):
        """Plot comparison of all metrics"""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [
            self.overall_results['overall_accuracy'],
            self.overall_results['overall_precision'],
            self.overall_results['overall_recall'],
            self.overall_results['overall_f1'],
            self.overall_results['overall_auc']
        ]
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        bars = ax.bar(metrics, values, color=colors, alpha=0.8)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Overall Metrics Comparison', fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{value:.3f}\n({value*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    def _plot_precision_recall_curve(self, ax):
        """Plot precision-recall curve"""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        y_true = self.overall_results['y_true']
        y_prob = self.overall_results['y_prob']
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        ax.plot(recall, precision, color='purple', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')
        ax.axhline(y=self.overall_results['overall_precision'], 
                  color='red', linestyle='--', alpha=0.5, label='Overall Precision')
        
        ax.set_xlabel('Recall', fontweight='bold')
        ax.set_ylabel('Precision', fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    
    def _plot_fold_confusion_summary(self, ax):
        """Plot confusion matrix summary across folds"""
        # Calculate average confusion matrix
        avg_cm = np.mean([r['confusion_matrix'] for r in self.results], axis=0)
        
        sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Greens', ax=ax,
                   cbar=False, square=True, linewidths=2, linecolor='black')
        
        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title('Average Confusion Matrix\n(Across Folds)', fontweight='bold')
        ax.set_xticklabels(['HC', 'AD'])
        ax.set_yticklabels(['HC', 'AD'])
    
    def _plot_performance_over_folds(self, ax):
        """Plot detailed performance metrics over folds"""
        folds = [r['fold'] for r in self.results]
        
        metrics = {
            'Accuracy': [r['accuracy'] for r in self.results],
            'Precision': [r['precision'] for r in self.results],
            'Recall': [r['recall'] for r in self.results],
            'F1-Score': [r['f1'] for r in self.results],
            'AUC': [r['auc'] for r in self.results]
        }
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
        
        for (metric_name, values), color in zip(metrics.items(), colors):
            ax.plot(folds, values, marker='o', linewidth=2, markersize=8,
                   label=metric_name, color=color, alpha=0.8)
            
            # Add mean line
            mean_value = np.mean(values)
            ax.axhline(y=mean_value, color=color, linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Fold', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title('Detailed Performance Metrics Across Folds', fontweight='bold', fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(folds)
    
    def _save_classification_report(self, timestamp):
        """Save detailed classification report"""
        output_dir = self.config.OUTPUT_DIR
        report_path = os.path.join(output_dir, f'cv_test_results_{timestamp}.txt')
        
        y_true = self.overall_results['y_true']
        y_pred = self.overall_results['y_pred']
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CROSS-VALIDATION TEST RESULTS\n")
            f.write("="*80 + "\n")
            f.write(f"\nTest Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Number of Folds: {self.config.K_FOLDS}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("OVERALL PERFORMANCE\n")
            f.write(f"{'='*80}\n")
            f.write(f"Accuracy:  {self.overall_results['overall_accuracy']:.4f} ({self.overall_results['overall_accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {self.overall_results['overall_precision']:.4f} ({self.overall_results['overall_precision']*100:.2f}%)\n")
            f.write(f"Recall:    {self.overall_results['overall_recall']:.4f} ({self.overall_results['overall_recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {self.overall_results['overall_f1']:.4f} ({self.overall_results['overall_f1']*100:.2f}%)\n")
            f.write(f"AUC:       {self.overall_results['overall_auc']:.4f} ({self.overall_results['overall_auc']*100:.2f}%)\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("PER-FOLD STATISTICS\n")
            f.write(f"{'='*80}\n")
            f.write(f"Accuracy:  {self.overall_results['avg_accuracy']:.4f} ± {self.overall_results['std_accuracy']:.4f}\n")
            f.write(f"F1-Score:  {self.overall_results['avg_f1']:.4f} ± {self.overall_results['std_f1']:.4f}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("CONFUSION MATRIX\n")
            f.write(f"{'='*80}\n")
            cm = self.overall_results['overall_cm']
            f.write(f"               Predicted\n")
            f.write(f"             HC    AD\n")
            f.write(f"Actual  HC  {cm[0,0]:4d} {cm[0,1]:4d}\n")
            f.write(f"        AD  {cm[1,0]:4d} {cm[1,1]:4d}\n")
            
            f.write(f"\n{'='*80}\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write(f"{'='*80}\n")
            f.write(classification_report(y_true, y_pred, target_names=['HC', 'AD']))
            
            f.write(f"\n{'='*80}\n")
            f.write("FOLD-BY-FOLD RESULTS\n")
            f.write(f"{'='*80}\n")
            for result in self.results:
                f.write(f"\nFold {result['fold']}:\n")
                f.write(f"  Accuracy:  {result['accuracy']:.4f}\n")
                f.write(f"  Precision: {result['precision']:.4f}\n")
                f.write(f"  Recall:    {result['recall']:.4f}\n")
                f.write(f"  F1-Score:  {result['f1']:.4f}\n")
                f.write(f"  AUC:       {result['auc']:.4f}\n")
        
        print(f"✅ Saved detailed report: {report_path}")


def main():
    """Main testing function"""
    print("="*80)
    print("CROSS-VALIDATION TESTING - TRAINED MODEL EVALUATION")
    print("="*80)
    
    # Initialize configuration
    config = Config()
    
    # Model path
    model_path = os.path.join(config.BASE_DIR, 'models', 'best_model.pth')
    
    # Create tester
    tester = CrossValidationTester(model_path, config)
    
    # Load model
    if not tester.load_trained_model():
        print("\n❌ Failed to load model. Please ensure the model is trained first.")
        print("Run: python src/eeg_alzheimer_detection.py")
        return None
    
    # Load dataset
    print(f"\n{'='*60}")
    print("LOADING TEST DATASET")
    print(f"{'='*60}")
    
    X, y, subjects = load_dataset(config.RAW_DATA_DIR, config)
    
    if X is None or y is None or subjects is None:
        print("\n❌ Failed to load dataset.")
        return None
    
    print(f"✅ Dataset loaded successfully")
    print(f"   Total samples: {len(X)}")
    print(f"   AD samples: {np.sum(y == 1)}")
    print(f"   HC samples: {np.sum(y == 0)}")
    
    # Perform cross-validation testing
    results = tester.perform_cross_validation_test(X, y, subjects)
    
    # Generate visualizations
    tester.generate_visualizations()
    
    # Print final summary
    print(f"\n{'='*80}")
    print("TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"✅ Model successfully validated on {config.K_FOLDS}-fold cross-validation")
    print(f"✅ Overall F1-Score: {tester.overall_results['overall_f1']:.4f} ({tester.overall_results['overall_f1']*100:.2f}%)")
    print(f"✅ Overall Accuracy: {tester.overall_results['overall_accuracy']:.4f} ({tester.overall_results['overall_accuracy']*100:.2f}%)")
    print(f"✅ Results saved to: {config.OUTPUT_DIR}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    try:
        results = main()
        if results is None:
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
