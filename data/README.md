# Dataset Information

## Overview

This project uses **Amazon 5-core review datasets** from the [Amazon Reviews dataset](https://nijianmo.github.io/amazon/index.html) (McAuley et al.).

**Note:** The raw dataset files are **NOT included** in this repository due to size constraints.

## Required Datasets

You need to download the following **5-core** datasets (gzipped JSON format):

1. **Electronics_5.json.gz**
2. **All_Beauty_5.json.gz**  
3. **Home_and_Kitchen_5.json.gz**

## Where to Download

Visit: [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html)

Navigate to the "Small" subsets section and download the 5-core versions for the three categories above.

## Dataset Placement

After downloading, place the `.json.gz` files in the `data/` directory of this repository:

```
data/
├── README.md (this file)
├── Electronics_5.json.gz
├── All_Beauty_5.json.gz
└── Home_and_Kitchen_5.json.gz
```

## Configuring the Script

The main script (`src/projectmilestone2.py`) contains hardcoded paths for Google Colab. You need to update these paths to point to your local `data/` directory.

**Lines to update** (around lines 52-54):

```python
# Original (Google Colab):
electronics_df = load_partial_json('/content/drive/MyDrive/EECS4414_project_dataset/Electronics_5.json.gz', 100000)
beauty_df      = load_partial_json('/content/drive/MyDrive/EECS4414_project_dataset/All_Beauty_5.json.gz', 100000)
home_df        = load_partial_json('/content/drive/MyDrive/EECS4414_project_dataset/Home_and_Kitchen_5.json.gz', 100000)

# Replace with local paths:
electronics_df = load_partial_json('data/Electronics_5.json.gz', 100000)
beauty_df      = load_partial_json('data/All_Beauty_5.json.gz', 100000)
home_df        = load_partial_json('data/Home_and_Kitchen_5.json.gz', 100000)
```

Or use the provided `src/run.py` entry point which handles path configuration automatically.

## Dataset Format

Each dataset is a gzipped JSON file with one review record per line. Key fields used:
- `reviewerID`: User identifier
- `asin`: Product identifier (Amazon Standard Identification Number)
- `unixReviewTime`: Review timestamp (Unix epoch)

## Citation

If you use these datasets, please cite:

```
Justifying recommendations using distantly-labeled reviews and fined-grained aspects
Jianmo Ni, Jiacheng Li, Julian McAuley
Empirical Methods in Natural Language Processing (EMNLP), 2019
```
