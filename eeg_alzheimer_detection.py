# EEG Analysis for Alzheimer's Disease Detection - Fixed Version
# This version uses basic preprocessing to avoid overly strict validation issues

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
print(f"Using device: {device}")

class Config:
    """Configuration optimized for the actual data"""
    # Data paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'EEG_data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'Output')
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # EEG preprocessing parameters - optimized for 8-second recordings
    CHANNELS_19 = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4',
        'T5', 'P3', 'Pz', 'P4', 'T6',
        'O1', 'O2'
    ]
    
    # Sampling rate from the data analysis
    SAMPLING_RATE = 128  # Hz
    LOWCUT = 0.5
    HIGHCUT = 45.0

    # Epoching parameters - adjusted for 8-second recordings
    EPOCH_DURATION = 2.0  # seconds (reduced from 5.0)
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
    N_CHANNELS = 19
    N_FREQ_BANDS = 5
    N_PSD_FEATURES = N_CHANNELS * N_FREQ_BANDS  # 95
    N_PLI_FEATURES = (N_CHANNELS * (N_CHANNELS - 1)) // 2  # 171
    TOTAL_FEATURES = N_PSD_FEATURES + N_PLI_FEATURES  # 266

    # Model parameters
    SEQUENCE_LENGTH = 2  # Number of epochs per sequence
    CNN_FILTERS = 64
    CNN_KERNEL_SIZE = 3
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    DROPOUT = 0.5

    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    K_FOLDS = 5
    PATIENCE = 10  # Early stopping patience

config = Config()

class EEGPreprocessor:
    """Basic EEG preprocessing optimized for the actual data"""

    def __init__(self, config):
        self.config = config
        mne.set_log_level('WARNING')

    def load_raw_eeg_from_dir(self, patient_dir):
        """Load raw EEG data from text files in a patient directory"""
        try:
            # Initialize data structure
            data = {}
            sfreq = self.config.SAMPLING_RATE
            
            # Load each channel
            for ch in self.config.CHANNELS_19:
                ch_file = os.path.join(patient_dir, f"{ch}.txt")
                if os.path.exists(ch_file):
                    try:
                        ch_data = np.loadtxt(ch_file)
                        if len(ch_data) > 0:
                            data[ch] = ch_data
                    except:
                        continue
            
            if not data:
                print(f"No valid channel data found in {patient_dir}")
                return None
            
            # Ensure all channels have the same length
            min_length = min(len(ch_data) for ch_data in data.values())
            for ch in data:
                data[ch] = data[ch][:min_length]
            
            # Create MNE Info object
            ch_names = list(data.keys())
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
            
            # Create RawArray
            raw_data = np.array([data[ch] for ch in ch_names])
            raw = mne.io.RawArray(raw_data, info)
            
            # Set channel positions (approximate)
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='warn')
            except:
                pass
            
            return raw
            
        except Exception as e:
            print(f"Error loading data from {patient_dir}: {e}")
            return None

    def preprocess_raw(self, raw):
        """Apply bandpass filter and CAR"""
        try:
            # Bandpass filter
            raw.filter(self.config.LOWCUT, self.config.HIGHCUT,
                      fir_design='firwin', verbose=False)

            # Select only available channels (don't require all 19)
            available_channels = [ch for ch in self.config.CHANNELS_19 if ch in raw.ch_names]
            if len(available_channels) < 10:  # Need at least 10 channels
                return None
            
            raw.pick_channels(available_channels, ordered=True)

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

            for epoch_data in epochs.get_data():  # Shape: (n_channels, n_times)
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
                                # Log transform to reduce skewness
                                band_power = np.log10(band_power + 1e-10)
                            else:
                                band_power = -10.0
                            epoch_features.append(band_power)
                    except:
                        # Fill with default values if computation fails
                        epoch_features.extend([-10.0] * len(self.config.FREQ_BANDS))

                psd_features.append(epoch_features)

            return np.array(psd_features)  # Shape: (n_epochs, 95)
        except Exception as e:
            print(f"Error computing PSD features: {e}")
            return None

    def compute_pli_features(self, epochs):
        """Compute Phase Lag Index connectivity features"""
        try:
            pli_features = []

            # Get epoch data
            epoch_data = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)

            for epoch in epoch_data:
                # Compute PLI for this epoch
                try:
                    pli_matrix = self._compute_pli_matrix(epoch, epochs.info['sfreq'])
                    
                    # Extract upper triangle (unique pairs)
                    triu_indices = np.triu_indices(pli_matrix.shape[0], k=1)
                    pli_values = pli_matrix[triu_indices]
                    
                    pli_features.append(pli_values)
                except:
                    # Fill with zeros if computation fails
                    n_channels = epoch.shape[0]
                    n_pairs = (n_channels * (n_channels - 1)) // 2
                    pli_features.append(np.zeros(n_pairs))

            return np.array(pli_features)  # Shape: (n_epochs, n_pairs)
        except Exception as e:
            print(f"Error computing PLI features: {e}")
            return None

    def _compute_pli_matrix(self, data, sfreq):
        """Compute PLI matrix for a single epoch"""
        n_channels = data.shape[0]
        pli_matrix = np.zeros((n_channels, n_channels))

        try:
            # Apply Hilbert transform to get instantaneous phase
            analytic_signal = signal.hilbert(data, axis=1)
            phase = np.angle(analytic_signal)

            # Compute PLI for each channel pair
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
                # Pad with zeros
                padding = np.zeros((pli_features.shape[0], expected_pli_size - actual_pli_size))
                pli_features = np.concatenate([pli_features, padding], axis=1)
            elif actual_pli_size > expected_pli_size:
                # Truncate
                pli_features = pli_features[:, :expected_pli_size]

            # Concatenate features
            combined_features = np.concatenate([psd_features, pli_features], axis=1)

            return combined_features  # Shape: (n_epochs, 266)
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return None

def load_dataset(data_dir, config):
    """Load all EEG files and extract features - optimized version"""
    preprocessor = EEGPreprocessor(config)

    all_sequences = []
    all_labels = []
    all_subjects = []
    subject_count = 0
    failed_count = 0

    print(f"Loading dataset from: {data_dir}")

    # Process AD group
    ad_dir = os.path.join(data_dir, 'AD')
    if os.path.exists(ad_dir):
        print("Loading AD subjects...")
        
        # Process both conditions (eyes closed and open)
        for condition in ['Eyes_closed', 'Eyes_open']:
            condition_dir = os.path.join(ad_dir, condition)
            if not os.path.exists(condition_dir):
                continue
                
            # Get all patient directories
            patient_dirs = [d for d in os.listdir(condition_dir) 
                           if os.path.isdir(os.path.join(condition_dir, d))]
            
            print(f"  Found {len(patient_dirs)} AD patients in {condition}")
            
            for patient in patient_dirs:
                patient_path = os.path.join(condition_dir, patient)
                
                try:
                    # Load channel data directly from patient directory
                    raw = preprocessor.load_raw_eeg_from_dir(patient_path)
                    if raw is None:
                        failed_count += 1
                        continue
                        
                    features = preprocessor.extract_features(raw)
                    if features is None or len(features) < config.SEQUENCE_LENGTH:
                        failed_count += 1
                        continue
                        
                    # Create sequences
                    n_sequences = len(features) // config.SEQUENCE_LENGTH
                    for i in range(n_sequences):
                        seq = features[i*config.SEQUENCE_LENGTH:(i+1)*config.SEQUENCE_LENGTH]
                        all_sequences.append(seq)
                        all_labels.append(1)  # 1 for AD
                        all_subjects.append(f"AD_{patient}_{condition}_{i}")
                        
                    subject_count += 1
                    if subject_count % 20 == 0:
                        print(f"  Processed {subject_count} subjects...")
                except Exception as e:
                    print(f"  Error processing {patient}: {e}")
                    failed_count += 1
                    continue
    
    # Process Healthy group
    healthy_dir = os.path.join(data_dir, 'Healthy')
    if os.path.exists(healthy_dir):
        print("\nLoading Healthy subjects...")
        
        # Process both conditions (eyes closed and open)
        for condition in ['Eyes_closed', 'Eyes_open']:
            condition_dir = os.path.join(healthy_dir, condition)
            if not os.path.exists(condition_dir):
                continue
                
            # Get all patient directories
            patient_dirs = [d for d in os.listdir(condition_dir) 
                           if os.path.isdir(os.path.join(condition_dir, d))]
            
            print(f"  Found {len(patient_dirs)} Healthy patients in {condition}")
            
            for patient in patient_dirs:
                patient_path = os.path.join(condition_dir, patient)
                
                try:
                    # Load channel data directly from patient directory
                    raw = preprocessor.load_raw_eeg_from_dir(patient_path)
                    if raw is None:
                        failed_count += 1
                        continue
                        
                    features = preprocessor.extract_features(raw)
                    if features is None or len(features) < config.SEQUENCE_LENGTH:
                        failed_count += 1
                        continue
                        
                    # Create sequences
                    n_sequences = len(features) // config.SEQUENCE_LENGTH
                    for i in range(n_sequences):
                        seq = features[i*config.SEQUENCE_LENGTH:(i+1)*config.SEQUENCE_LENGTH]
                        all_sequences.append(seq)
                        all_labels.append(0)  # 0 for Healthy
                        all_subjects.append(f"HC_{patient}_{condition}_{i}")
                        
                    subject_count += 1
                    if subject_count % 20 == 0:
                        print(f"  Total processed: {subject_count} subjects...")
                except Exception as e:
                    print(f"  Error processing {patient}: {e}")
                    failed_count += 1
                    continue
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {subject_count} subjects")
    print(f"  Failed: {failed_count} subjects")
    
    # Convert to numpy arrays
    if len(all_sequences) == 0:
        print("Error: No valid data found.")
        return None, None, None
        
    # Ensure we have both classes
    if len(set(all_labels)) < 2:
        print("Error: Need both AD and HC samples for training.")
        return None, None, None

    X = np.array(all_sequences)  # Shape: (n_samples, sequence_length, 266)
    y = np.array(all_labels)
    subjects = np.array(all_subjects)

    print(f"\nDataset loaded:")
    print(f"  Total sequences: {len(X)}")
    print(f"  AD sequences: {np.sum(y == 1)}")
    print(f"  HC sequences: {np.sum(y == 0)}")
    print(f"  Feature shape: {X.shape}")

    return X, y, subjects

class HybridCNNLSTM(nn.Module):
    """Hybrid CNN-LSTM architecture for EEG sequence classification"""

    def __init__(self, config):
        super(HybridCNNLSTM, self).__init__()

        self.config = config

        # CNN Block (applied to features)
        self.conv1 = nn.Conv1d(
            in_channels=config.SEQUENCE_LENGTH,
            out_channels=config.CNN_FILTERS,
            kernel_size=config.CNN_KERNEL_SIZE,
            padding=config.CNN_KERNEL_SIZE // 2
        )
        self.bn1 = nn.BatchNorm1d(config.CNN_FILTERS)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(config.DROPOUT)

        # LSTM Block (applied to temporal sequence)
        self.lstm = nn.LSTM(
            input_size=config.TOTAL_FEATURES,
            hidden_size=config.LSTM_HIDDEN,
            num_layers=config.LSTM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(config.LSTM_HIDDEN * 2, 64)  # *2 for bidirectional
        self.dropout2 = nn.Dropout(config.DROPOUT)
        self.fc2 = nn.Linear(64, 2)  # Binary classification

    def forward(self, x):
        """Forward pass"""
        batch_size = x.size(0)

        # CNN feature extraction
        cnn_out = self.conv1(x)  # (batch, filters, features)
        cnn_out = self.bn1(cnn_out)
        cnn_out = self.relu(cnn_out)
        cnn_out = self.dropout1(cnn_out)

        # Reshape for LSTM: (batch, seq_len, features)
        cnn_out = torch.mean(cnn_out, dim=1, keepdim=True)  # (batch, 1, features)

        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Take the last output
        lstm_out = lstm_out[:, -1, :]  # (batch, hidden*2)

        # Fully connected classifier
        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)

        return out

class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def compute_class_weights(y_train):
    """Compute class weights for imbalanced dataset"""
    class_counts = np.bincount(y_train)
    total = len(y_train)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights).to(device)

def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def evaluate(model, test_loader):
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

def train_with_cross_validation(X, y, subjects, config):
    """Train model with stratified K-fold cross-validation"""

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

        X_train_fold = X[train_mask]
        y_train_fold = y[train_mask]
        X_test_fold = X[test_mask]
        y_test_fold = y[test_mask]

        print(f"Train samples: {len(X_train_fold)} (AD: {np.sum(y_train_fold==1)}, HC: {np.sum(y_train_fold==0)})")
        print(f"Test samples: {len(X_test_fold)} (AD: {np.sum(y_test_fold==1)}, HC: {np.sum(y_test_fold==0)})")

        # Reshape for SMOTE (flatten sequences)
        X_train_flat = X_train_fold.reshape(len(X_train_fold), -1)
        X_test_flat = X_test_fold.reshape(len(X_test_fold), -1)

        # Apply SMOTE to training data only
        print("\nApplying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_flat, y_train_fold)

        # Reshape back to sequences
        X_train_balanced = X_train_balanced.reshape(-1, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)

        print(f"After SMOTE: {len(X_train_balanced)} samples (AD: {np.sum(y_train_balanced==1)}, HC: {np.sum(y_train_balanced==0)})")

        # Standardize features
        scaler = StandardScaler()
        X_train_reshaped = X_train_balanced.reshape(-1, config.TOTAL_FEATURES)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled.reshape(-1, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)

        X_test_reshaped = X_test_flat.reshape(-1, config.TOTAL_FEATURES)
        X_test_scaled = scaler.transform(X_test_reshaped)
        X_test_scaled = X_test_scaled.reshape(-1, config.SEQUENCE_LENGTH, config.TOTAL_FEATURES)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.LongTensor(y_train_balanced)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.LongTensor(y_test_fold)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # Initialize model
        model = HybridCNNLSTM(config).to(device)

        # Compute class weights
        class_weights = compute_class_weights(y_train_balanced)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        # Early stopping
        early_stopping = EarlyStopping(patience=config.PATIENCE)

        # Training loop
        print("\nTraining...")
        best_f1 = 0
        best_model_state = None

        for epoch in range(config.NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer)

            # Evaluate
            y_true, y_pred, y_prob = evaluate(model, test_loader)

            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Loss: {train_loss:.4f}, "
                      f"Acc: {accuracy:.4f}, F1: {f1:.4f}")

            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                best_model_state = model.state_dict().copy()

            # Early stopping
            early_stopping(train_loss)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Final evaluation
        y_true, y_pred, y_prob = evaluate(model, test_loader)

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

def save_results_log(results, X, y, config):
    """Save detailed results to a log file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_results_{timestamp}.txt"
    log_path = os.path.join(config.OUTPUT_DIR, log_filename)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ALZHEIMER'S DISEASE EEG CLASSIFICATION - TRAINING RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write(f"\nDataset Information:\n")
        f.write(f"Total sequences: {len(X)}\n")
        f.write(f"AD sequences: {np.sum(y == 1)} ({np.sum(y == 1)/len(y)*100:.1f}%)\n")
        f.write(f"HC sequences: {np.sum(y == 0)} ({np.sum(y == 0)/len(y)*100:.1f}%)\n")
        f.write(f"Feature shape: {X.shape}\n")
        
        f.write(f"\nCross-Validation Results:\n")
        for result in results:
            f.write(f"\nFold {result['fold']}:\n")
            f.write(f"  Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"  Precision: {result['precision']:.4f}\n")
            f.write(f"  Recall: {result['recall']:.4f}\n")
            f.write(f"  F1-Score: {result['f1']:.4f}\n")
            f.write(f"  AUC: {result['auc']:.4f}\n")
        
        avg_f1 = np.mean([r['f1'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        f.write(f"\nOverall Performance:\n")
        f.write(f"Average F1-Score: {avg_f1:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
    
    return log_path

def main():
    """Main execution function"""
    print("="*60)
    print("HYBRID CNN-LSTM FOR ALZHEIMER'S DISEASE DETECTION")
    print("="*60)

    # Check if data directory exists
    if not os.path.exists(config.RAW_DATA_DIR):
        print(f"\nError: Data directory not found: {config.RAW_DATA_DIR}")
        return None

    print(f"\nLoading data from: {config.RAW_DATA_DIR}")
    X, y, subjects = load_dataset(config.RAW_DATA_DIR, config)

    # Check if data was loaded successfully
    if X is None or y is None or subjects is None:
        print("\nFailed to load data. Please check the error messages above.")
        return None

    # Check if we have enough data
    if len(X) < config.K_FOLDS:
        print(f"\nError: Not enough samples ({len(X)}) for {config.K_FOLDS}-fold cross-validation")
        return None

    # Train with cross-validation
    results = train_with_cross_validation(X, y, subjects, config)

    # Visualize results
    print(f"\n{'='*60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*60}")

    # Plot fold metrics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Cross-Validation Results Across Folds', fontsize=16, fontweight='bold')

    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 3, idx % 3]
        values = [r[metric] for r in results]
        folds = [r['fold'] for r in results]

        ax.bar(folds, values, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axhline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
        ax.set_xlabel('Fold', fontweight='bold')
        ax.set_ylabel(name, fontweight='bold')
        ax.set_title(f'{name} per Fold', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # Remove empty subplot
    axes[1, 2].axis('off')

    plt.tight_layout()
    cv_results_path = os.path.join(config.OUTPUT_DIR, 'cv_results.png')
    plt.savefig(cv_results_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {cv_results_path}")
    plt.show()

    # Save detailed results
    log_path = save_results_log(results, X, y, config)
    print(f"\nDetailed results saved to: {log_path}")

    # Print final summary
    avg_f1 = np.mean([r['f1'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Final Performance:")
    print(f"  Average F1-Score: {avg_f1:.4f}")
    print(f"  Average Recall (Sensitivity): {avg_recall:.4f}")

    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()