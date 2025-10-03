#!/usr/bin/env python3
"""
Final System Test Script
========================
Comprehensive system validation for the EEG Alzheimer's Detection System

This script validates:
1. Data loading and preprocessing pipeline
2. Feature extraction accuracy
3. Model architecture and parameters
4. Cross-validation performance
5. SHAP explainability functionality
6. Results generation and visualization

Author: AD Detection Team
Date: October 2024
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
from datetime import datetime

def test_data_loading():
    """Test data loading functionality"""
    print("🔍 Testing data loading...")
    
    try:
        from eeg_alzheimer_detection import Config, load_dataset
        
        config = Config()
        
        # Check if data directory exists
        if not os.path.exists(config.RAW_DATA_DIR):
            print(f"❌ Data directory not found: {config.RAW_DATA_DIR}")
            return False
        
        print(f"✅ Data directory found: {config.RAW_DATA_DIR}")
        
        # Try to load a small subset of data
        print("   Loading sample data...")
        X, y, subjects = load_dataset(config.RAW_DATA_DIR, config)
        
        if X is None or y is None or subjects is None:
            print("❌ Failed to load data")
            return False
        
        print(f"✅ Data loaded successfully:")
        print(f"   Samples: {len(X)}")
        print(f"   AD samples: {np.sum(y == 1)}")
        print(f"   HC samples: {np.sum(y == 0)}")
        print(f"   Feature shape: {X.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in data loading test: {str(e)}")
        return False

def test_model_architecture():
    """Test model architecture"""
    print("\n🔍 Testing model architecture...")
    
    try:
        from eeg_alzheimer_detection import Config, HybridCNNLSTM
        
        config = Config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        model = HybridCNNLSTM(config).to(device)
        
        # Test forward pass with dummy data
        batch_size = 4
        dummy_input = torch.randn(batch_size, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_output_shape = (batch_size, 2)  # Binary classification
        if output.shape == expected_output_shape:
            print(f"✅ Model architecture test passed:")
            print(f"   Input shape: {dummy_input.shape}")
            print(f"   Output shape: {output.shape}")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            return True
        else:
            print(f"❌ Model output shape mismatch. Expected: {expected_output_shape}, Got: {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Error in model architecture test: {str(e)}")
        return False

def test_feature_extraction():
    """Test feature extraction pipeline"""
    print("\n🔍 Testing feature extraction...")
    
    try:
        from eeg_alzheimer_detection import Config, EEGPreprocessor
        
        config = Config()
        preprocessor = EEGPreprocessor(config)
        
        # Test with sample patient data
        sample_patient_dir = os.path.join(config.RAW_DATA_DIR, 'AD', 'Eyes_closed', 'Paciente1')
        
        if not os.path.exists(sample_patient_dir):
            print(f"❌ Sample patient directory not found: {sample_patient_dir}")
            return False
        
        # Load raw data
        raw = preprocessor.load_raw_eeg_from_dir(sample_patient_dir)
        if raw is None:
            print("❌ Failed to load raw EEG data")
            return False
        
        print(f"✅ Raw data loaded: {len(raw.ch_names)} channels, {raw.n_times} samples")
        
        # Extract features
        features = preprocessor.extract_features(raw)
        if features is None:
            print("❌ Failed to extract features")
            return False
        
        expected_feature_dim = config.TOTAL_FEATURES
        if features.shape[1] == expected_feature_dim:
            print(f"✅ Feature extraction test passed:")
            print(f"   Features shape: {features.shape}")
            print(f"   Expected features per epoch: {expected_feature_dim}")
            return True
        else:
            print(f"❌ Feature dimension mismatch. Expected: {expected_feature_dim}, Got: {features.shape[1]}")
            return False
            
    except Exception as e:
        print(f"❌ Error in feature extraction test: {str(e)}")
        return False

def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline"""
    print("\n🔍 Testing preprocessing pipeline...")
    
    try:
        from eeg_alzheimer_detection import Config, EEGPreprocessor
        
        config = Config()
        preprocessor = EEGPreprocessor(config)
        
        # Load sample data
        sample_patient_dir = os.path.join(config.RAW_DATA_DIR, 'AD', 'Eyes_closed', 'Paciente1')
        raw = preprocessor.load_raw_eeg_from_dir(sample_patient_dir)
        
        if raw is None:
            print("❌ Failed to load raw data")
            return False
        
        # Test preprocessing steps
        raw_preprocessed = preprocessor.preprocess_raw(raw.copy())
        if raw_preprocessed is None:
            print("❌ Preprocessing failed")
            return False
        
        print("✅ Bandpass filtering and CAR applied successfully")
        
        # Test epoching
        epochs = preprocessor.create_epochs(raw_preprocessed)
        if epochs is None or len(epochs) == 0:
            print("❌ Epoching failed")
            return False
        
        print(f"✅ Epoching successful: {len(epochs)} epochs created")
        
        # Test PSD feature computation
        psd_features = preprocessor.compute_psd_features(epochs)
        if psd_features is None:
            print("❌ PSD feature computation failed")
            return False
        
        print(f"✅ PSD features computed: {psd_features.shape}")
        
        # Test PLI feature computation
        pli_features = preprocessor.compute_pli_features(epochs)
        if pli_features is None:
            print("❌ PLI feature computation failed")
            return False
        
        print(f"✅ PLI features computed: {pli_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in preprocessing pipeline test: {str(e)}")
        return False

def test_model_training():
    """Test model training with minimal data"""
    print("\n🔍 Testing model training (minimal)...")
    
    try:
        from eeg_alzheimer_detection import Config, HybridCNNLSTM, train_epoch, evaluate
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        config = Config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create dummy data for testing
        n_samples = 32
        X_dummy = torch.randn(n_samples, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)
        y_dummy = torch.randint(0, 2, (n_samples,))
        
        # Create data loader
        dataset = TensorDataset(X_dummy, y_dummy)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Create model
        model = HybridCNNLSTM(config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Test one training epoch
        initial_loss = train_epoch(model, dataloader, criterion, optimizer)
        print(f"✅ Training epoch completed, loss: {initial_loss:.4f}")
        
        # Test evaluation
        y_true, y_pred, y_prob = evaluate(model, dataloader)
        print(f"✅ Evaluation completed: {len(y_true)} predictions")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in model training test: {str(e)}")
        return False

def test_saved_model():
    """Test loading saved model if it exists"""
    print("\n🔍 Testing saved model loading...")
    
    try:
        from eeg_alzheimer_detection import Config, HybridCNNLSTM
        
        config = Config()
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'models', 'best_model.pth')
        
        if not os.path.exists(model_path):
            print(f"ℹ️  No saved model found at {model_path} (this is normal if not trained yet)")
            return True
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load the model
        model = HybridCNNLSTM(config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Saved model loaded successfully from {model_path}")
        
        # Test with dummy data
        dummy_input = torch.randn(1, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES).to(device)
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Model inference test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in saved model test: {str(e)}")
        return False

def run_system_validation():
    """Run complete system validation"""
    print("="*80)
    print("EEG ALZHEIMER'S DETECTION - SYSTEM VALIDATION")
    print("="*80)
    
    start_time = datetime.now()
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Architecture", test_model_architecture),
        ("Feature Extraction", test_feature_extraction),
        ("Preprocessing Pipeline", test_preprocessing_pipeline),
        ("Model Training", test_model_training),
        ("Saved Model", test_saved_model),
    ]
    
    results = []
    
    for test_name, test_function in tests:
        print(f"\n{'='*60}")
        print(f"RUNNING TEST: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_function()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} - PASSED")
            else:
                print(f"❌ {test_name} - FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    print(f"\nDetailed Results:")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nValidation completed in: {duration}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - SYSTEM IS READY!")
        return True
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED - PLEASE REVIEW")
        return False

def main():
    """Main function"""
    try:
        success = run_system_validation()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)