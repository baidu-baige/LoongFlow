# -*- coding: utf-8 -*-
"""
LLM Generated Code
"""

import aifc
import io
import os
import zipfile
from typing import Tuple

import numpy as np
import pandas as pd

# Define the data paths
BASE_DATA_PATH = "/root/workspace/evolux/output/mlebench/the-icml-2013-whale-challenge-right-whale-redux/prepared/public"
OUTPUT_DATA_PATH = "output/5872ea81-cdbf-4bc6-8da8-7d0e82d40021/2/executor/output"

# For type hinting, DT is assumed to be pandas DataFrame or Series
DT = pd.DataFrame | pd.Series


def load_data(validation_mode: bool = False) -> Tuple[DT, DT, DT, DT]:
    """
    Loads, splits, and returns the initial datasets for the Right Whale Detection challenge.
    
    Args:
        validation_mode: Controls the data loading behavior.
            - False (default): Load the complete dataset.
            - True: Load a small subset (≤50 rows).

    Returns:
        Tuple[DT, DT, DT, DT]: (X, y, X_test, test_ids)
    """
    train_zip_path = os.path.join(BASE_DATA_PATH, "train2.zip")
    test_zip_path = os.path.join(BASE_DATA_PATH, "test2.zip")

    def extract_audio_data(zip_path, limit=None):
        data_list = []
        info_list = []

        with zipfile.ZipFile(zip_path, 'r') as z:
            # Filter for .aif files and exclude directories
            files = sorted([f for f in z.namelist() if f.endswith('.aif') and not z.getinfo(f).is_dir()])

            if limit:
                # Sample a representative subset across the file list
                step = max(1, len(files) // limit)
                files = files[::step][:limit]

            for f_name in files:
                with z.open(f_name) as f:
                    file_content = f.read()
                    with aifc.open(io.BytesIO(file_content)) as af:
                        params = af.getparams()
                        # params: (nchannels, sampwidth, framerate, nframes, comptype, compname)
                        n_frames = params.nframes
                        frames = af.readframes(n_frames)

                        # AIFF is Big-Endian. EDA confirmed 16-bit (sampwidth=2)
                        if params.sampwidth == 2:
                            audio = np.frombuffer(frames, dtype='>i2').astype(np.float32)
                        else:
                            audio = np.frombuffer(frames, dtype='b').astype(np.float32)

                        # Handle multi-channel (though EDA suggests mono)
                        if params.nchannels > 1:
                            audio = audio.reshape(-1, params.nchannels).mean(axis=1)

                        # Target length consistency: 3800 samples
                        target_len = 3800
                        if len(audio) > target_len:
                            audio = audio[:target_len]
                        elif len(audio) < target_len:
                            audio = np.pad(audio, (0, target_len - len(audio)), 'constant')

                        data_list.append(audio)
                        info_list.append(os.path.basename(f_name))

        return np.vstack(data_list), info_list

    # Load Training Data
    train_limit = 50 if validation_mode else None
    train_audio, train_filenames = extract_audio_data(train_zip_path, train_limit)

    # X and X_test must have identical feature structure
    X = pd.DataFrame(train_audio)

    # Extract labels: _1.aif -> 1 (whale), _0.aif -> 0 (noise)
    y = pd.Series([1 if f.endswith('_1.aif') else 0 for f in train_filenames], name='label')

    # Load Test Data
    test_limit = 50 if validation_mode else None
    test_audio, test_filenames = extract_audio_data(test_zip_path, test_limit)

    X_test = pd.DataFrame(test_audio)
    test_ids = pd.Series(test_filenames, name='clip')

    # Final consistency checks (Row alignment ensured by extract_audio_data implementation)
    return X, y, X_test, test_ids
