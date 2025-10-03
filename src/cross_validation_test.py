"""
Cross-Validation Test for EEG Alzheimer's Detection
==================================================
This script tests the trained model using cross-validation on different data splits.
"""

import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

import numpy as np
import pandas as pd
import mne
from scipy import signal

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    """Configuration for cross-validation test"""
    # Data paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up to project root
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')  # Training data
    VALIDATION_DATA_DIR = os.path.join(BASE_DIR, 'data', 'ds004504')  # Independent validation data
    OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pth')
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # EEG preprocessing parameters
    CHANNELS_19 = [
        'Fp1', 'Fp2', 'F1', 'F2', 'F3', 'F4', 'F7', 'F8', 'Fz',
        'C3', 'C4', 'Cz',
        'P3', 'P4', 'Pz',
        'T3', 'T4', 'T5', 'T6',
        'O1', 'O2'
    ]
    
    SAMPLING_RATE = 128  # Hz
    LOWCUT = 0.5
    HIGHCUT = 45.0

    # Epoching parameters - adjusted for SET file data
    EPOCH_DURATION = 2.0  # seconds
    EPOCH_OVERLAP = 0.5  # 50% overlap

    # Frequency bands for PSD
    FREQ_BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 45)
    }

    # Feature dimensions
    N_CHANNELS = 21
    N_FREQ_BANDS = 5
    N_PSD_FEATURES = N_CHANNELS * N_FREQ_BANDS
    N_PLI_FEATURES = (N_CHANNELS * (N_CHANNELS - 1)) // 2
    TOTAL_FEATURES = N_PSD_FEATURES + N_PLI_FEATURES

    # Model parameters
    SEQUENCE_LENGTH = 2
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    DROPOUT = 0.5

    # Cross-validation parameters
    K_FOLDS = 5
    BATCH_SIZE = 16

config = Config()

class EEGPreprocessor:
    """EEG preprocessing for SET files"""

    def __init__(self, config):
        self.config = config
        mne.set_log_level('WARNING')

    def load_set_file(self, set_file_path):
        """Load EEG data from SET file"""
        try:
            # Load the SET file using MNE
            raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
            
            # Get available channels that match our target channels
            available_channels = [ch for ch in self.config.CHANNELS_19 if ch in raw.ch_names]
            
            if len(available_channels) < 10:  # Need at least 10 channels
                print(f"Warning: Only {len(available_channels)} channels available in {set_file_path}")
                return None
            
            # Select available channels
            raw.pick_channels(available_channels, ordered=True)
            
            return raw
            
        except Exception as e:
            print(f"Error loading {set_file_path}: {e}")
            return None

    def preprocess_raw(self, raw):
        """Apply bandpass filter and CAR"""
        try:
            # Bandpass filter
            raw.filter(self.config.LOWCUT, self.config.HIGHCUT,
                      fir_design='firwin', verbose=False)

            # Apply Common Average Reference (CAR)
            raw.set_eeg_reference('average', projection=False, verbose=False)

            return raw
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def create_epochs(self, raw):
        """Create overlapping epochs from continuous data"""
        try:
            epoch_length = self.config.EPOCH_DURATION
            overlap = self.config.EPOCH_OVERLAP

            # Create fixed-length events
            events = mne.make_fixed_length_events(
                raw,
                duration=epoch_length * (1 - overlap)
            )

            # Create epochs
            epochs = mne.Epochs(
                raw,
                events,
                tmin=0,
                tmax=epoch_length,
                baseline=None,
                preload=True,
                verbose=False
            )

            return epochs
        except Exception as e:
            print(f"Error creating epochs: {e}")
            return None

    def compute_psd_features(self, epochs):
        """Compute Power Spectral Density features for all frequency bands"""
        try:
            psd_features = []

            for epoch_data in epochs.get_data():
                epoch_features = []

                for ch_idx in range(epoch_data.shape[0]):
                    ch_data = epoch_data[ch_idx]

                    try:
                        # Compute PSD using Welch's method
                        freqs, psd = signal.welch(
                            ch_data,
                            fs=epochs.info['sfreq'],
                            nperseg=min(256, len(ch_data))
                        )

                        # Extract mean power in each frequency band
                        for band_name, (fmin, fmax) in self.config.FREQ_BANDS.items():
                            idx_band = np.logical_and(freqs >= fmin, freqs <= fmax)
                            if np.any(idx_band):
                                band_power = np.mean(psd[idx_band])
                                band_power = np.log10(band_power + 1e-10)
                            else:
                                band_power = -10.0
                            epoch_features.append(band_power)
                    except:
                        epoch_features.extend([-10.0] * len(self.config.FREQ_BANDS))

                # Pad or truncate to expected PSD feature size
                expected_psd_size = self.config.N_PSD_FEATURES
                if len(epoch_features) < expected_psd_size:
                    epoch_features.extend([-10.0] * (expected_psd_size - len(epoch_features)))
                elif len(epoch_features) > expected_psd_size:
                    epoch_features = epoch_features[:expected_psd_size]

                psd_features.append(epoch_features)

            return np.array(psd_features)
        except Exception as e:
            print(f"Error computing PSD features: {e}")
            return None

    def compute_pli_features(self, epochs):
        """Compute Phase Lag Index connectivity features"""
        try:
            pli_features = []
            epoch_data = epochs.get_data()

            for epoch in epoch_data:
                try:
                    pli_matrix = self._compute_pli_matrix(epoch, epochs.info['sfreq'])
                    triu_indices = np.triu_indices(pli_matrix.shape[0], k=1)
                    pli_values = pli_matrix[triu_indices]
                    pli_features.append(pli_values)
                except:
                    n_channels = epoch.shape[0]
                    n_pairs = (n_channels * (n_channels - 1)) // 2
                    pli_features.append(np.zeros(n_pairs))

            return np.array(pli_features)
        except Exception as e:
            print(f"Error computing PLI features: {e}")
            return None

    def _compute_pli_matrix(self, data, sfreq):
        """Compute PLI matrix for a single epoch"""
        n_channels = data.shape[0]
        pli_matrix = np.zeros((n_channels, n_channels))

        try:
            analytic_signal = signal.hilbert(data, axis=1)
            phase = np.angle(analytic_signal)

            for i in range(n_channels):
                for j in range(i + 1, n_channels):
                    try:
                        phase_diff = phase[i] - phase[j]
                        pli_value = np.abs(np.mean(np.sign(phase_diff)))
                        
                        if np.isnan(pli_value) or np.isinf(pli_value):
                            pli_value = 0.0
                            
                        pli_matrix[i, j] = pli_value
                        pli_matrix[j, i] = pli_value
                    except:
                        pli_matrix[i, j] = 0.0
                        pli_matrix[j, i] = 0.0
        except:
            pass

        return pli_matrix

    def extract_features(self, raw):
        """Complete feature extraction pipeline"""
        try:
            # Preprocess
            raw = self.preprocess_raw(raw)
            if raw is None:
                return None

            # Create epochs
            epochs = self.create_epochs(raw)
            if epochs is None or len(epochs) == 0:
                return None

            # Extract PSD features
            psd_features = self.compute_psd_features(epochs)
            if psd_features is None:
                return None

            # Extract PLI features
            pli_features = self.compute_pli_features(epochs)
            if pli_features is None:
                return None

            # Pad or truncate PLI features to match expected size
            expected_pli_size = self.config.N_PLI_FEATURES
            actual_pli_size = pli_features.shape[1]
            
            if actual_pli_size < expected_pli_size:
                padding = np.zeros((pli_features.shape[0], expected_pli_size - actual_pli_size))
                pli_features = np.concatenate([pli_features, padding], axis=1)
            elif actual_pli_size > expected_pli_size:
                pli_features = pli_features[:, :expected_pli_size]

            # Concatenate features
            combined_features = np.concatenate([psd_features, pli_features], axis=1)
            
            # Ensure features have the correct size
            if combined_features.shape[1] != self.config.TOTAL_FEATURES:
                print(f"Warning: Feature size mismatch. Expected {self.config.TOTAL_FEATURES}, got {combined_features.shape[1]}")
                if combined_features.shape[1] < self.config.TOTAL_FEATURES:
                    # Pad with zeros
                    padding = np.zeros((combined_features.shape[0], self.config.TOTAL_FEATURES - combined_features.shape[1]))
                    combined_features = np.concatenate([combined_features, padding], axis=1)
                else:
                    # Truncate
                    combined_features = combined_features[:, :self.config.TOTAL_FEATURES]

            return combined_features
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None

class HybridCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM architecture for EEG sequence classification"""

    def __init__(self, config):
        super(HybridCNNLSTM, self).__init__()

        self.config = config

        # CNN Block
        self.conv1 = nn.Conv1d(
            in_channels=config.SEQUENCE_LENGTH,
            out_channels=config.CNN_FILTERS,
            kernel_size=config.CNN_KERNEL_SIZE,
            padding=config.CNN_KERNEL_SIZE // 2
        )
        self.bn1 = nn.BatchNorm1d(config.CNN_FILTERS)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(config.DROPOUT)

        # LSTM Block
        self.lstm = nn.LSTM(
            input_size=config.TOTAL_FEATURES,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(config.LSTM_HIDDEN * 2, 64)
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        """Forward pass"""
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_out = self.conv1(x)
        cnn_out = self.bn1(cnn_out)
        cnn_out = self.relu(cnn_out)
        cnn_out = self.dropout1(cnn_out)

        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]

        # Fully connected classifier
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out

def load_bids_dataset(data_dir, config):
    """Load BIDS format dataset from ds004504 for independent validation"""
    preprocessor = EEGPreprocessor(config)

    all_sequences = []
    all_labels = []
    all_subjects = []
    
    print(f"Loading BIDS dataset from: {data_dir}")
    
    # Check if ds004504 directory exists
    if not os.path.exists(data_dir):
        print(f"Error: BIDS directory not found at {data_dir}")
        return None, None, None
    
    # Read participants.tsv to get group labels
    participants_file = os.path.join(data_dir, 'participants.tsv')
    if not os.path.exists(participants_file):
        print(f"Error: participants.tsv not found at {participants_file}")
        return None, None, None
    
    # Load participant information
    import pandas as pd
    participants_df = pd.read_csv(participants_file, sep='\t')
    
    # Create label mapping: A=Alzheimer's, C=Control, F=Frontotemporal (exclude F for binary classification)
    subject_labels = {}
    for _, row in participants_df.iterrows():
        if row['Group'] == 'A':  # Alzheimer's
            subject_labels[row['participant_id']] = 1
        elif row['Group'] == 'C':  # Control
            subject_labels[row['participant_id']] = 0
        # Skip F (Frontotemporal) for binary AD vs Control classification
    
    print(f"Found {len(subject_labels)} subjects for binary classification (AD vs Control)")
    
    # Process each subject
    processed_count = 0
    failed_count = 0
    
    for subject_id, label in subject_labels.items():
        subject_dir = os.path.join(data_dir, subject_id, 'eeg')
        
        if not os.path.exists(subject_dir):
            failed_count += 1
            continue
        
        # Look for .set files
        set_files = [f for f in os.listdir(subject_dir) if f.endswith('.set')]
        
        if not set_files:
            failed_count += 1
            continue
        
        # Process the first .set file found
        set_file_path = os.path.join(subject_dir, set_files[0])
        
        try:
            # Load the SET file using MNE
            raw = mne.io.read_raw_eeglab(set_file_path, preload=True, verbose=False)
            
            if raw is None:
                failed_count += 1
                continue
            
            # Extract features
            features = preprocessor.extract_features(raw)
            if features is None or len(features) < config.SEQUENCE_LENGTH:
                failed_count += 1
                continue
            
            # Create sequences with consistent length
            n_sequences = len(features) // config.SEQUENCE_LENGTH
            if n_sequences > 0:
                for i in range(n_sequences):
                    seq = features[i*config.SEQUENCE_LENGTH:(i+1)*config.SEQUENCE_LENGTH]
                    # Ensure sequence has correct shape
                    if seq.shape == (config.SEQUENCE_LENGTH, config.TOTAL_FEATURES):
                        all_sequences.append(seq)
                        all_labels.append(label)
                        all_subjects.append(f"{subject_id}_{i}")
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count} subjects...")
                
        except Exception as e:
            print(f"  Error processing {subject_id}: {e}")
            failed_count += 1
            continue
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed_count} subjects")
    print(f"  Failed: {failed_count} subjects")
    
    if len(all_sequences) == 0:
        print("Error: No valid data found.")
        return None, None, None
    
    # Check shape consistency
    expected_shape = (config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)
    valid_sequences = []
    valid_labels = []
    valid_subjects = []
    
    for seq, label, subject in zip(all_sequences, all_labels, all_subjects):
        if seq.shape == expected_shape:
            valid_sequences.append(seq)
            valid_labels.append(label)
            valid_subjects.append(subject)
    
    if len(valid_sequences) == 0:
        print("Error: No sequences with correct shape found.")
        return None, None, None
    
    # Ensure we have both classes
    if len(set(valid_labels)) < 2:
        print("Error: Need both AD and Control samples for validation.")
        return None, None, None

    X = np.array(valid_sequences)
    y = np.array(valid_labels)
    subjects = np.array(valid_subjects)

    print(f"\nBIDS dataset loaded:")
    print(f"  Total sequences: {len(X)}")
    print(f"  AD sequences: {np.sum(y == 1)}")
    print(f"  Control sequences: {np.sum(y == 0)}")
    print(f"  Feature shape: {X.shape}")

    return X, y, subjects

def load_raw_text_dataset(data_dir, config):
    """Load all text-based EEG files and extract features"""
    from eeg_alzheimer_detection import load_dataset
    return load_dataset(data_dir, config)

def load_model(model_path, config):
    """Load the trained model"""
    try:
        model = HybridCNNLSTM(config).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✓ Model loaded successfully from {model_path}")
        print(f"  Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
        
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            outputs = model(batch_X)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def validate_model_on_independent_data(X, y, subjects, model_path, config):
    """Validate the trained model on independent BIDS dataset"""
    
    # Load the trained model
    model = load_model(model_path, config)
    if model is None:
        return None

    print(f"\n{'='*60}")
    print(f"INDEPENDENT VALIDATION ON BIDS DATASET")
    print(f"{'='*60}")

    print(f"Validation samples: {len(X)} (AD: {np.sum(y==1)}, Control: {np.sum(y==0)})")

    # Standardize features (use global statistics from the data)
    scaler = StandardScaler()
    X_flat = X.reshape(-1, config.TOTAL_FEATURES)
    X_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_scaled.reshape(-1, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y)

    # Create data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Evaluate
    y_true, y_pred, y_prob = evaluate_model(model, dataloader)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred)

    print(f"\nIndependent Validation Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'n_samples': len(y_true),
        'n_ad': np.sum(y_true == 1),
        'n_control': np.sum(y_true == 0)
    }

    return result

def cross_validate_model(X, y, subjects, model_path, config):
    """Perform cross-validation using the trained model"""
    
    # Load the trained model
    model = load_model(model_path, config)
    if model is None:
        return None

    # Subject-level stratified K-fold
    unique_subjects = np.unique(subjects)
    subject_labels = np.array([y[subjects == subj][0] for subj in unique_subjects])

    skf = StratifiedKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(unique_subjects, subject_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{config.K_FOLDS}")
        print(f"{'='*60}")

        # Get subject splits
        train_subjects = unique_subjects[train_idx]
        test_subjects = unique_subjects[test_idx]

        # Get sample indices for this fold
        train_mask = np.isin(subjects, train_subjects)
        test_mask = np.isin(subjects, test_subjects)

        X_test_fold = X[test_mask]
        y_test_fold = y[test_mask]

        print(f"Test samples: {len(X_test_fold)} (AD: {np.sum(y_test_fold==1)}, HC: {np.sum(y_test_fold==0)})")

        # Standardize features (use global statistics)
        scaler = StandardScaler()
        X_flat = X.reshape(-1, config.TOTAL_FEATURES)
        scaler.fit(X_flat)
        
        X_test_flat = X_test_fold.reshape(-1, config.TOTAL_FEATURES)
        X_test_scaled = scaler.transform(X_test_flat)
        X_test_scaled = X_test_scaled.reshape(-1, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)

        # Convert to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test_fold)

        # Create data loader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Evaluate
        y_true, y_pred, y_prob = evaluate_model(model, test_loader)

        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0

        cm = confusion_matrix(y_true, y_pred)

        print(f"\nFold {fold+1} Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)

        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        })

    # Print average results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_precision = np.mean([r['precision'] for r in fold_results])
    avg_recall = np.mean([r['recall'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_auc = np.mean([r['auc'] for r in fold_results])

    print(f"Average Accuracy:  {avg_accuracy:.4f} ± {np.std([r['accuracy'] for r in fold_results]):.4f}")
    print(f"Average Precision: {avg_precision:.4f} ± {np.std([r['precision'] for r in fold_results]):.4f}")
    print(f"Average Recall:    {avg_recall:.4f} ± {np.std([r['recall'] for r in fold_results]):.4f}")
    print(f"Average F1-Score:  {avg_f1:.4f} ± {np.std([r['f1'] for r in fold_results]):.4f}")
    print(f"Average AUC:       {avg_auc:.4f} ± {np.std([r['auc'] for r in fold_results]):.4f}")

    return fold_results

def save_validation_results(result, config):
    """Save independent validation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = os.path.join(config.OUTPUT_DIR, f"independent_validation_results_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write("Independent Validation Results on BIDS Dataset (ds004504)\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Dataset Information:\n")
        f.write(f"  Total samples: {result['n_samples']}\n")
        f.write(f"  AD samples: {result['n_ad']}\n")
        f.write(f"  Control samples: {result['n_control']}\n\n")
        
        f.write(f"Performance Metrics:\n")
        f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
        f.write(f"  Precision: {result['precision']:.4f}\n")
        f.write(f"  Recall: {result['recall']:.4f}\n")
        f.write(f"  F1-Score: {result['f1']:.4f}\n")
        f.write(f"  AUC: {result['auc']:.4f}\n\n")
        
        f.write(f"Confusion Matrix:\n")
        f.write(f"{result['confusion_matrix']}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot confusion matrix
    ax1 = axes[0]
    cm = result['confusion_matrix']
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=16, fontweight='bold')
    
    ax1.set_xlabel('Predicted Label', fontweight='bold')
    ax1.set_ylabel('True Label', fontweight='bold')
    ax1.set_title('Confusion Matrix\n(Independent Validation)', fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Control', 'AD'])
    ax1.set_yticklabels(['Control', 'AD'])
    
    # Plot performance metrics
    ax2 = axes[1]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    values = [result['accuracy'], result['precision'], result['recall'], result['f1'], result['auc']]
    
    bars = ax2.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax2.set_title('Performance Metrics\n(Independent Validation)', fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    plot_file = os.path.join(config.OUTPUT_DIR, f"independent_validation_plot_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_file}")

def save_cv_results(results, config):
    """Save cross-validation results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = os.path.join(config.OUTPUT_DIR, f"cv_test_results_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write("Cross-Validation Test Results\n")
        f.write("="*50 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for result in results:
            f.write(f"Fold {result['fold']}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1-Score: {result['f1']:.4f}\n")
            f.write(f"  AUC: {result['auc']:.4f}\n\n")
        
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        f.write(f"Average F1-Score: {avg_f1:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [r[metric] for r in results]
        folds = [r['fold'] for r in results]
        
        if i < 3:  # First 3 metrics in first subplot
            axes[0].plot(folds, values, 'o-', label=name, linewidth=2, markersize=6)
        else:  # Last 2 metrics in second subplot
            axes[1].plot(folds, values, 'o-', label=name, linewidth=2, markersize=6)
    
    for ax in axes:
        ax.set_xlabel('Fold')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[0].set_title('Accuracy, Precision, Recall')
    axes[1].set_title('F1-Score, AUC')
    
    plt.suptitle('Cross-Validation Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = os.path.join(config.OUTPUT_DIR, f"cv_test_plot_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {plot_file}")

def main():
    print("="*80)
    print("CROSS-VALIDATION TEST FOR EEG ALZHEIMER'S DETECTION")
    print("BIDS Dataset Validation (ds004504)")
    print("="*80)
    print()
    
    print("This script is designed to validate the model on the BIDS dataset ds004504.")
    print("However, the actual EEG data files are stored using Git Annex and need to")
    print("be downloaded separately.")
    print()
    print("Current status: The .set files in data/ds004504/ are Git Annex pointers")
    print("(small files pointing to the actual data stored elsewhere).")
    print()
    print("To download the actual BIDS data:")
    print("1. Install git-annex: https://git-annex.branchable.com/install/")
    print("2. Navigate to data/ds004504/")
    print("3. Run: git annex get .")
    print()
    print("Alternatively, download from OpenNeuro:")
    print("https://openneuro.org/datasets/ds004504/versions/1.0.8")
    print()
    print("Once the data is available, run:")
    print("python src/bids_validation.py")
    print()
    print("For immediate independent validation using available data:")
    print("python src/independent_validation.py")
    print("(This uses a holdout subset of the training data for validation)")
    print("="*80)

if __name__ == "__main__":
    main()
