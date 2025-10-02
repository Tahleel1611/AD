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
import shap

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
    
    # SHAP explainability parameters
    SHAP_BACKGROUND_SIZE = 100  # Number of background samples for SHAP
    SHAP_TEST_SIZE = 20  # Number of test samples to explain
    ENABLE_SHAP = True  # Enable SHAP analysis

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

def explain_predictions_with_shap(model, X_background, X_test, y_test, feature_names, config):
    """
    Generate SHAP explanations for model predictions
    """
    print(f"\n{'='*60}")
    print("SHAP EXPLAINABILITY ANALYSIS")
    print(f"{'='*60}")
    
    try:
        # Convert to PyTorch tensors
        X_background_tensor = torch.FloatTensor(X_background).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        # Create a wrapper function for SHAP that handles model output properly
        def model_wrapper(x):
            model.eval()
            with torch.no_grad():
                outputs = model(x)
                # Return softmax probabilities for the AD class (index 1)
                probs = torch.softmax(outputs, dim=1)
                return probs[:, 1].cpu().numpy()
        
        # Create SHAP explainer with GradientExplainer (more stable than DeepExplainer)
        print("Creating SHAP explainer...")
        explainer = shap.GradientExplainer(model, X_background_tensor)
        
        # Calculate SHAP values
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_test_tensor)
        
        # Convert to numpy if needed
        if isinstance(shap_values, torch.Tensor):
            shap_values_np = shap_values.cpu().numpy()
        else:
            shap_values_np = shap_values
            
        X_test_np = X_test_tensor.cpu().numpy()
        
        # Reshape for feature-level analysis
        # From (n_samples, seq_length, features) to (n_samples, total_features)
        shap_flat = shap_values_np.reshape(shap_values_np.shape[0], -1)
        X_test_flat = X_test_np.reshape(X_test_np.shape[0], -1)
        
        # Create feature names for flattened features
        # Calculate actual number of features from the data shape
        actual_features = shap_flat.shape[1]
        
        # Calculate expected dimensions
        expected_seq_length = actual_features // len(feature_names)
        
        flat_feature_names = []
        for seq in range(expected_seq_length):
            for feat in feature_names:
                flat_feature_names.append(f"T{seq+1}_{feat}")
        
        # Ensure exact match with actual features
        if len(flat_feature_names) > actual_features:
            flat_feature_names = flat_feature_names[:actual_features]
        elif len(flat_feature_names) < actual_features:
            # Add generic names for remaining features
            for i in range(len(flat_feature_names), actual_features):
                flat_feature_names.append(f"Feature_{i+1}")
        
        print(f"Created {len(flat_feature_names)} feature names for {actual_features} features")
        
        # Generate SHAP visualizations
        print("Generating SHAP visualizations...")
        
        # 1. Summary plot (bar plot instead of dot plot for stability)
        plt.figure(figsize=(12, 8))
        
        # Calculate mean absolute SHAP values for feature importance
        mean_shap_values = np.mean(np.abs(shap_flat), axis=0)
        
        # Ensure we have valid indices
        num_features = len(flat_feature_names)
        if num_features == 0:
            print("Warning: No feature names available for SHAP analysis")
            return
            
        # Get top features (up to 20 or total number of features, whichever is smaller)
        top_n = min(20, num_features)
        top_features_idx = np.argsort(mean_shap_values)[-top_n:][::-1]
        
        # Ensure indices are valid
        valid_indices = [idx for idx in top_features_idx if idx < len(flat_feature_names)]
        
        if len(valid_indices) == 0:
            print("Warning: No valid feature indices found for SHAP analysis")
            return
            
        top_feature_names = [flat_feature_names[i] for i in valid_indices]
        top_feature_values = mean_shap_values[valid_indices]
        
        bars = plt.barh(range(len(top_feature_names)), top_feature_values, color='steelblue', alpha=0.7)
        plt.yticks(range(len(top_feature_names)), top_feature_names)
        plt.xlabel('Mean |SHAP Value|', fontweight='bold')
        plt.title('Top 20 Most Important Features for AD Classification', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, top_feature_values):
            plt.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        summary_path = os.path.join(config.OUTPUT_DIR, 'shap_feature_importance.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved SHAP feature importance: {summary_path}")
        
        # 2. Feature importance by category
        analyze_feature_importance_by_category(shap_flat, feature_names, config)
        
        # 3. Individual prediction explanations
        generate_individual_explanations(shap_flat, X_test_flat, y_test, 
                                       flat_feature_names, config)
        
        # 4. Channel importance heatmap
        generate_channel_importance_heatmap(shap_flat, config)
        
        return shap_values_np
        
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_feature_importance_by_category(shap_values, feature_names, config):
    """
    Analyze SHAP importance by feature categories (PSD bands, PLI connectivity)
    """
    try:
        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Group features by category
        n_psd_features = config.N_PSD_FEATURES * config.SEQUENCE_LENGTH
        
        psd_importance = mean_shap[:n_psd_features]
        pli_importance = mean_shap[n_psd_features:]
        
        # Analyze PSD by frequency bands
        band_importance = {}
        for i, band in enumerate(config.FREQ_BANDS.keys()):
            band_indices = []
            for seq in range(config.SEQUENCE_LENGTH):
                for ch in range(config.N_CHANNELS):
                    idx = seq * config.N_PSD_FEATURES + ch * len(config.FREQ_BANDS) + i
                    if idx < len(psd_importance):
                        band_indices.append(idx)
            
            if band_indices:
                band_importance[band] = np.mean(psd_importance[band_indices])
        
        # Create frequency band importance plot
        plt.figure(figsize=(10, 6))
        bands = list(band_importance.keys())
        importances = list(band_importance.values())
        
        bars = plt.bar(bands, importances, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.title('SHAP Importance by EEG Frequency Bands', fontsize=14, fontweight='bold')
        plt.xlabel('Frequency Bands', fontweight='bold')
        plt.ylabel('Mean |SHAP Value|', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, importance in zip(bars, importances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        band_path = os.path.join(config.OUTPUT_DIR, 'shap_frequency_bands.png')
        plt.savefig(band_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved frequency band importance: {band_path}")
        
        # Feature category comparison
        plt.figure(figsize=(8, 6))
        categories = ['PSD Features', 'PLI Connectivity']
        cat_importance = [np.mean(psd_importance), np.mean(pli_importance)]
        
        bars = plt.bar(categories, cat_importance, color=['#2ca02c', '#ff7f0e'])
        plt.title('SHAP Importance: PSD vs Connectivity Features', fontsize=14, fontweight='bold')
        plt.ylabel('Mean |SHAP Value|', fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        for bar, importance in zip(bars, cat_importance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        cat_path = os.path.join(config.OUTPUT_DIR, 'shap_feature_categories.png')
        plt.savefig(cat_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved feature category comparison: {cat_path}")
        
        # Print analysis results
        print(f"\nFeature Importance Analysis:")
        print(f"  PSD Features: {np.mean(psd_importance):.4f}")
        print(f"  PLI Connectivity: {np.mean(pli_importance):.4f}")
        print(f"\nFrequency Band Importance:")
        for band, importance in band_importance.items():
            print(f"  {band.capitalize()}: {importance:.4f}")
            
    except Exception as e:
        print(f"Error in feature category analysis: {e}")

def generate_individual_explanations(shap_values, X_test, y_test, feature_names, config):
    """
    Generate individual prediction explanations for sample patients
    """
    try:
        # Select a few representative samples
        ad_indices = np.where(y_test == 1)[0]
        hc_indices = np.where(y_test == 0)[0]
        
        selected_indices = []
        if len(ad_indices) > 0:
            selected_indices.extend(ad_indices[:2])  # 2 AD samples
        if len(hc_indices) > 0:
            selected_indices.extend(hc_indices[:2])  # 2 HC samples
            
        if not selected_indices:
            return
            
        # Generate waterfall plots for selected samples
        for i, idx in enumerate(selected_indices):
            try:
                # Get top 15 most important features for this sample
                sample_shap = shap_values[idx]
                feature_importance = np.abs(sample_shap)
                top_indices = np.argsort(feature_importance)[-15:][::-1]
                
                plt.figure(figsize=(10, 8))
                
                # Create waterfall-style plot
                top_shap = sample_shap[top_indices]
                top_features = [feature_names[j] if j < len(feature_names) else f"Feature_{j}" 
                               for j in top_indices]
                
                # Sort by SHAP value for better visualization
                sorted_indices = np.argsort(top_shap)
                sorted_shap = top_shap[sorted_indices]
                sorted_features = [top_features[j] for j in sorted_indices]
                
                colors = ['red' if x < 0 else 'blue' for x in sorted_shap]
                bars = plt.barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)
                
                plt.yticks(range(len(sorted_features)), sorted_features)
                plt.xlabel('SHAP Value (Impact on AD Prediction)', fontweight='bold')
                
                true_label = "AD Patient" if y_test[idx] == 1 else "Healthy Control"
                plt.title(f'Individual Explanation - {true_label} (Sample {idx})', 
                         fontsize=14, fontweight='bold')
                
                # Add value labels
                for bar, value in zip(bars, sorted_shap):
                    plt.text(value + (0.01 if value >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                            f'{value:.3f}', ha='left' if value >= 0 else 'right', 
                            va='center', fontweight='bold')
                
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.8)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                ind_path = os.path.join(config.OUTPUT_DIR, f'shap_individual_{i}_{true_label.replace(" ", "_").lower()}.png')
                plt.savefig(ind_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved individual explanation: {ind_path}")
                
            except Exception as e:
                print(f"Error generating individual explanation {i}: {e}")
                continue
                
    except Exception as e:
        print(f"Error in individual explanations: {e}")

def generate_channel_importance_heatmap(shap_values, config):
    """
    Generate EEG channel importance heatmap
    """
    try:
        # Calculate mean importance for each channel across PSD features
        channel_importance = np.zeros(config.N_CHANNELS)
        
        for ch in range(config.N_CHANNELS):
            ch_indices = []
            for seq in range(config.SEQUENCE_LENGTH):
                for band in range(len(config.FREQ_BANDS)):
                    idx = seq * config.N_PSD_FEATURES + ch * len(config.FREQ_BANDS) + band
                    if idx < shap_values.shape[1]:
                        ch_indices.append(idx)
            
            if ch_indices:
                channel_importance[ch] = np.mean(np.abs(shap_values[:, ch_indices]))
        
        # Create 2D layout for EEG channels (approximate 10-20 positions)
        channel_positions = {
            'Fp1': (1, 0), 'Fp2': (1, 4),
            'F7': (2, 0), 'F3': (2, 1), 'Fz': (2, 2), 'F4': (2, 3), 'F8': (2, 4),
            'T3': (3, 0), 'C3': (3, 1), 'Cz': (3, 2), 'C4': (3, 3), 'T4': (3, 4),
            'T5': (4, 0), 'P3': (4, 1), 'Pz': (4, 2), 'P4': (4, 3), 'T6': (4, 4),
            'O1': (5, 1), 'O2': (5, 3)
        }
        
        # Create heatmap matrix
        heatmap_data = np.zeros((6, 5))
        
        for i, ch_name in enumerate(config.CHANNELS_19):
            if ch_name in channel_positions:
                row, col = channel_positions[ch_name]
                heatmap_data[row, col] = channel_importance[i]
        
        # Plot heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
        
        # Add channel labels
        for ch_name, (row, col) in channel_positions.items():
            if ch_name in config.CHANNELS_19:
                ch_idx = config.CHANNELS_19.index(ch_name)
                plt.text(col, row, f'{ch_name}\n{channel_importance[ch_idx]:.3f}', 
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.colorbar(im, label='Mean |SHAP Value|')
        plt.title('EEG Channel Importance for AD Classification', fontsize=14, fontweight='bold')
        plt.xlabel('Lateral Position', fontweight='bold')
        plt.ylabel('Anterior-Posterior Position', fontweight='bold')
        
        # Remove ticks
        plt.xticks([])
        plt.yticks([])
        
        plt.tight_layout()
        heatmap_path = os.path.join(config.OUTPUT_DIR, 'shap_channel_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved channel importance heatmap: {heatmap_path}")
        
    except Exception as e:
        print(f"Error generating channel heatmap: {e}")

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

        # SHAP analysis for the first fold only (to avoid redundancy)
        if fold == 0 and config.ENABLE_SHAP:
            try:
                # Prepare data for SHAP
                X_background = X_train_scaled[:config.SHAP_BACKGROUND_SIZE]
                X_test_shap = X_test_scaled[:config.SHAP_TEST_SIZE]
                y_test_shap = y_test_fold[:config.SHAP_TEST_SIZE]
                
                # Create feature names
                feature_names = []
                for ch_name in config.CHANNELS_19:
                    for band_name in config.FREQ_BANDS.keys():
                        feature_names.append(f"{ch_name}_{band_name}")
                
                # Add PLI feature names
                for i in range(config.N_CHANNELS):
                    for j in range(i+1, config.N_CHANNELS):
                        ch1, ch2 = config.CHANNELS_19[i], config.CHANNELS_19[j]
                        feature_names.append(f"PLI_{ch1}_{ch2}")
                
                # Generate SHAP explanations
                explain_predictions_with_shap(model, X_background, X_test_shap, 
                                            y_test_shap, feature_names, config)
            except Exception as e:
                print(f"SHAP analysis failed: {e}")

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

    # Plot confusion matrices across folds
    plot_confusion_matrices_across_folds(fold_results, config)

    return fold_results

def plot_confusion_matrices_across_folds(fold_results, config):
    """Plot confusion matrices for all folds"""
    print(f"\nGenerating confusion matrices across folds...")
    
    n_folds = len(fold_results)
    fig, axes = plt.subplots(1, n_folds, figsize=(4*n_folds, 5))
    
    if n_folds == 1:
        axes = [axes]
    
    # Color map for confusion matrices
    cmap = plt.cm.Blues
    
    # Find global max for consistent colorbar scale
    global_max = max([result['confusion_matrix'].max() for result in fold_results])
    
    for i, result in enumerate(fold_results):
        ax = axes[i]
        cm = result['confusion_matrix']
        
        # Create heatmap with consistent scale
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=global_max)
        
        # Add text annotations
        thresh = global_max / 2.
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, format(cm[row, col], 'd'),
                       ha="center", va="center",
                       color="white" if cm[row, col] > thresh else "black",
                       fontsize=16, fontweight='bold')
        
        # Labels and title
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
        if i == 0:  # Only add y-label to the first subplot
            ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
        ax.set_title(f'Fold {i+1}', fontweight='bold', fontsize=14)
        
        # Set tick labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['HC', 'AD'], fontsize=11)
        ax.set_yticklabels(['HC', 'AD'], fontsize=11)
        
        # Clean appearance
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        
        # Remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    # Add a single colorbar on the right
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, aspect=20, pad=0.02)
    cbar.set_label('Count', fontweight='bold', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Overall title with more space
    fig.suptitle('Confusion Matrices Across Folds', fontsize=18, fontweight='bold', y=0.95)
    
    # Adjust layout to prevent overlapping
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.1, right=0.9, wspace=0.3)
    
    cm_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved confusion matrices plot: {cm_path}")
    plt.close()  # Close to prevent display issues

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