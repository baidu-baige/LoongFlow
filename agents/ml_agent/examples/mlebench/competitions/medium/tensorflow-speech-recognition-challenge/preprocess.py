import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Any, List

# Task-adaptive type definitions
X = pd.DataFrame      # Input feature matrix (contains 'file_path')
y = pd.Series         # Target vector (labels)
X_out = np.ndarray    # Processed features (N, 128, 101)
y_out = np.ndarray    # Processed targets (N,)

# Constants for Speech Processing
SR = 16000
N_SAMPLES = 16000
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 1024
F_MIN = 20.0
F_MAX = 8000.0

def process_single_audio(
    file_path: str, 
    augment: bool, 
    noise_data: List[torch.Tensor], 
    seed: int
) -> np.ndarray:
    """
    Processes a single audio file: loads, pads/trims, augments, and converts to Mel-spectrogram.
    """
    # Set seeds for reproducibility within parallel workers
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    # 1. Load Waveform
    try:
        waveform, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        # Propagate implementation-critical errors
        raise RuntimeError(f"Failed to load {file_path}: {e}")

    # Resample if necessary
    if sample_rate != SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SR)
        waveform = resampler(waveform)
    
    waveform = waveform.squeeze()

    # 2. Handle Duration (Pad/Trim to 1.0s)
    if len(waveform) < N_SAMPLES:
        waveform = torch.nn.functional.pad(waveform, (0, N_SAMPLES - len(waveform)))
    elif len(waveform) > N_SAMPLES:
        waveform = waveform[:N_SAMPLES]

    # 3. Waveform Augmentations (Only for training)
    if augment:
        # Random time shift (+/- 100ms -> +/- 1600 samples)
        shift = np.random.randint(-1600, 1601)
        waveform = torch.roll(waveform, shifts=shift, dims=0)
        
        # Addition of background noise (scaled by 0.1)
        if noise_data:
            noise = noise_data[np.random.randint(len(noise_data))]
            if len(noise) >= N_SAMPLES:
                start = np.random.randint(0, len(noise) - N_SAMPLES + 1)
                noise_segment = noise[start : start + N_SAMPLES]
                waveform = waveform + 0.1 * noise_segment

    # 4. Mel-Spectrogram Calculation
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=F_MIN,
        f_max=F_MAX,
        power=2.0
    )
    mel_spec = mel_transform(waveform)

    # 5. Log Scaling
    # Use log10 or natural log; adding a small epsilon to avoid log(0)
    log_mel_spec = torch.log(mel_spec + 1e-9)

    # 6. SpecAugment (Only for training)
    if augment:
        # Frequency masking
        freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        log_mel_spec = freq_mask(log_mel_spec)
        # Time masking
        time_mask = torchaudio.transforms.TimeMasking(time_mask_param=20)
        log_mel_spec = time_mask(log_mel_spec)

    return log_mel_spec.numpy()

def preprocess(
    X_train: X,
    y_train: y,
    X_val: X,
    y_val: y,
    X_test: X
) -> Tuple[X_out, y_out, X_out, y_out, X_out]:
    """
    Transforms raw audio paths into Log-Mel Spectrogram tensors.
    """
    print("Starting preprocessing: converting waveforms to Mel-spectrograms.")

    # Step 1: Fit Label Encoder on Training Targets
    # Ensure fixed alphabetical sorting for consistency across pipeline
    le = LabelEncoder()
    le.fit(y_train)
    y_train_processed = le.transform(y_train)
    y_val_processed = le.transform(y_val)
    print(f"Encoded {len(le.classes_)} classes.")

    # Step 2: Prepare Background Noise for Augmentation
    # Noise is located in train/audio/_background_noise_
    BASE_DATA_PATH = "/mnt/pfs/loongflow/devmachine/2-05/evolux/output/mlebench/tensorflow-speech-recognition-challenge/prepared/public"
    noise_dir = os.path.join(BASE_DATA_PATH, "train/audio/_background_noise_")
    noise_data = []
    if os.path.exists(noise_dir):
        print("Loading background noise files for augmentation...")
        for f in sorted(os.listdir(noise_dir)):
            if f.endswith('.wav'):
                p = os.path.join(noise_dir, f)
                w, _ = torchaudio.load(p)
                noise_data.append(w.squeeze())
    
    # Step 3: Transform Audio Clips in Parallel
    def run_parallel_processing(df, augment, noise_list, desc):
        print(f"Processing {desc} set ({len(df)} samples)...")
        paths = df['file_path'].values
        # Using 36 cores as per hardware context
        results = Parallel(n_jobs=36)(
            delayed(process_single_audio)(path, augment, noise_list, i) 
            for i, path in enumerate(paths)
        )
        # Stack into (N, 128, 101)
        return np.stack(results).astype(np.float32)

    X_train_processed = run_parallel_processing(X_train, True, noise_data, "train")
    X_val_processed = run_parallel_processing(X_val, False, [], "validation")
    X_test_processed = run_parallel_processing(X_test, False, [], "test")

    # Step 4: Final Validation
    for name, data in [("X_train", X_train_processed), ("X_val", X_val_processed), ("X_test", X_test_processed)]:
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError(f"Output {name} contains NaN or Infinity values.")
        if data.shape[1:] != (128, 101):
            raise ValueError(f"Output {name} has incorrect shape: {data.shape}")

    print("Preprocessing complete. Shapes:")
    print(f"  Train: {X_train_processed.shape}, {y_train_processed.shape}")
    print(f"  Val:   {X_val_processed.shape}, {y_val_processed.shape}")
    print(f"  Test:  {X_test_processed.shape}")

    return X_train_processed, y_train_processed, X_val_processed, y_val_processed, X_test_processed