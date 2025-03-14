# Data for EEG-based REM Sleep Detection

This directory contains the data used for EEG-based REM sleep detection and mood prediction.

## Data Organization

The data is organized into subdirectories:

- `raw/`: Raw EEG datasets and sleep-mood data
- `processed/`: Processed data ready for feature extraction
- `features/`: Extracted features for model training

## Datasets

### Sleep-EDF Dataset

The Sleep-EDF dataset contains EEG recordings with sleep stage annotations. It is downloaded from PhysioNet and stored in `raw/sleep-edf/`.

Key files:
- `*-PSG.edf`: Polysomnography recordings containing EEG channels
- `*-Hypnogram.edf`: Sleep stage annotations

### Sleep-Cassette Dataset

The Sleep-Cassette dataset contains additional EEG recordings with sleep stage annotations. It is also downloaded from PhysioNet and stored in `raw/sleep-cassette/`.

Key files:
- `*-PSG.edf`: Polysomnography recordings containing EEG channels
- `*-Hypnogram.edf`: Sleep stage annotations

### Sleep-Mood Dataset

The Sleep-Mood dataset is a synthetic dataset generated for this project. It contains sleep metrics and corresponding mood ratings. It is stored in `raw/sleep-mood/`.

Key file:
- `sleep_mood_dataset.csv`: CSV file containing sleep metrics and mood ratings

## Processed Data

The processed data is organized by dataset:

- `processed/sleep-edf/`: Processed Sleep-EDF data
- `processed/sleep-cassette/`: Processed Sleep-Cassette data
- `processed/sleep-mood/`: Processed Sleep-Mood data

Each processed file contains extracted features and sleep stage labels.

## Features

The extracted features are stored in the `features/` directory:

- `rem_detection_features.joblib`: Features for REM sleep detection
- `mood_prediction_features.joblib`: Features for mood prediction

These files contain train/test splits and selected features for model training.

## Downloading Data

To download the datasets, use the `download_datasets.py` script in the `src/data` directory:

```bash
python src/data/download_datasets.py
```

This will download all datasets and store them in the appropriate directories.

## Processing Data

To process the raw data, use the `process_data.py` script in the `src/data` directory:

```bash
python src/data/process_data.py
```

This will process all datasets and store the results in the `processed/` directory.

## Extracting Features

To extract features from the processed data, use the `extract_features.py` script in the `src/features` directory:

```bash
python src/features/extract_features.py
```

This will extract features and store them in the `features/` directory. 