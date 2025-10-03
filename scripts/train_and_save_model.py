#!/usr/bin/env python3
"""
Train and Save Model Script
===========================
Train the EEG Alzheimer's detection model and save it for later use.

This script:
1. Loads and preprocesses the EEG data
2. Trains the hybrid CNN-LSTM model
3. Saves the best performing model
4. Generates training reports and visualizations

Author: AD Detection Team
Date: October 2024
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Main training function"""
    print("="*80)
    print("EEG ALZHEIMER'S DETECTION - MODEL TRAINING")
    print("="*80)
    
    try:
        # Import the main training module
        from eeg_alzheimer_detection import main as run_training
        
        print("Starting model training with cross-validation...")
        results = run_training()
        
        if results:
            print("\n" + "="*80)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            # Print summary
            import numpy as np
            avg_f1 = np.mean([r['f1'] for r in results])
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            avg_recall = np.mean([r['recall'] for r in results])
            avg_precision = np.mean([r['precision'] for r in results])
            
            print(f"Final Performance Summary:")
            print(f"  Average F1-Score:  {avg_f1:.4f}")
            print(f"  Average Accuracy:  {avg_accuracy:.4f}")
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  Average Recall:    {avg_recall:.4f}")
            
            print(f"\nModel saved successfully!")
            print(f"Ready for cross-validation testing.")
            
            return True
        else:
            print("❌ Training failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)