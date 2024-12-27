# Task-Aware Image Restoration (TAIR)

## Overview

The **Task-Aware Image Restoration (TAIR)** is a research project focusing on dynamically balancing restoration loss and task loss in image restoration tasks. The aim is to improve the synergy between image restoration and downstream task performance.

### Prerequisites
1. Python 3.10 or later
2. NVIDIA CUDA Toolkit version 12.2 (or compatible GPU drivers)
3. `pip install -r requirements.txt`

### Step

1. Create degraded dataset
    1. `cd data` and follow the instruction in `README.md` to download the datasets.
    2. `cd ..` and run `python 0_create_degraded_dataset.py -s [src_dir] -d [degraded_type]` to create degraded dataset.
2. Train the clear model