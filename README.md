# Predict Spike Counts with Linear-Gaussian Model

This repository contains a Python script to predict spike counts using a Linear-Gaussian model. The script downloads data, processes it, creates a design matrix, predicts spike counts, and plots the actual vs predicted spike counts.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)
- [Explanation](#explanation)
  - [Download Data](#download-data)
  - [Create Design Matrix](#create-design-matrix)
  - [Predict Spike Counts](#predict-spike-counts)
  - [Plot Results](#plot-results)
- [How to Run the Code](#how-to-run-the-code)
- [Expected Output](#expected-output)

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib
- Requests

You can install the necessary packages using pip:

```bash
pip install numpy scipy matplotlib requests
```

## Usage

Save the script as `spike_prediction.py` and run it:

```bash
python spike_prediction.py
```

## Explanation

### Download Data

The script begins by importing necessary libraries and defining constants for the filename, URL, expected MD5 checksum, cell number, and the number of timepoints to keep.

The `download_data` function checks if the data file already exists. If not, it downloads the file from the specified URL and verifies its integrity using an MD5 checksum.

### Create Design Matrix

The `make_design_matrix` function creates a time-lagged design matrix from the stimulus intensity vector. This matrix is used for linear regression to predict spike counts.

### Predict Spike Counts

The `predict_spike_counts_lg` function performs the following steps:

1. **Create the complete design matrix**: Uses the stimulus data to generate the design matrix.
2. **Obtain the MLE weights (θ̂)**: Calculates the maximum likelihood estimate weights using the ordinary least squares method.
3. **Compute ŷ = Xθ̂**: Uses the design matrix and the MLE weights to compute the predicted spike counts.

### Plot Results

The `plot_spikes_with_prediction` function plots the actual spike counts against the predicted spike counts over time. This visualization helps in assessing the model's performance.

## How to Run the Code

1. Make sure you have the necessary libraries installed:

    ```bash
    pip install numpy scipy matplotlib requests
    ```

2. Save the script as a `.py` file, for example, `spike_prediction.py`.

3. Run the script:

    ```bash
    python spike_prediction.py
    ```

## Expected Output

When you run the script, it will download the data (if not already present), predict spike counts, and plot the actual vs predicted spike counts.

You should see a plot with the x-axis representing time and the y-axis representing spike counts. The plot will display both actual spike counts (as points) and predicted spike counts (as a line).

## Code

Here is the full script:

```python
import os
import hashlib
import requests
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Constants
FNAME = "RGCdata.mat"
URL = "https://osf.io/mzujs/download"
EXPECTED_MD5 = "1b2977453020bce5319f2608c94d38d0"
CELLNUM = 2
KEEP_TIMEPOINTS = 20000

def download_data(fname, url, expected_md5):
    """Download the data file if not already present."""
    if not os.path.isfile(fname):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.ConnectionError:
            print("!!! Failed to download data !!!")
            return False
        else:
            if hashlib.md5(response.content).hexdigest() != expected_md5:
                print("!!! Data download appears corrupted !!!")
                return False
            with open(fname, "wb") as fid:
                fid.write(response.content)
    return True

def make_design_matrix(stim, d=25):
    """Create time-lag design matrix from stimulus intensity vector."""
    padded_stim = np.concatenate([np.zeros(d - 1), stim])
    T = len(stim)
    X = np.zeros((T, d))
    for t in range(T):
        X[t] = padded_stim[t:t + d]
    return X

def predict_spike_counts_lg(stim, spikes, d=25):
    """Compute a vector of predicted spike counts given the stimulus."""
    y = spikes
    constant = np.ones_like(y)
    X = np.column_stack([constant, make_design_matrix(stim)])
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    yhat = X @ theta
    return yhat

def plot_spikes_with_prediction(spikes, predicted_counts, dt_stim):
    """Plot actual vs predicted spike counts."""
    time = np.arange(len(spikes)) * dt_stim
    plt.figure(figsize=(15, 6))
    plt.plot(time, spikes, label='Actual Spikes', marker='o', linestyle='None', markersize=4)
    plt.plot(time, predicted_counts, label='Predicted Spikes', linestyle='-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Spike Counts')
    plt.title('Actual vs Predicted Spike Counts')
    plt.legend()
    plt.show()

def main():
    # Download data
    if not download_data(FNAME, URL, EXPECTED_MD5):
        return

    # Load data
    data = loadmat(FNAME)
    dt_stim = data['dtStim'].item()
    stim = data['Stim'].squeeze()
    spikes = data['SpCounts'][:, CELLNUM]

    # Trim data to keep desired timepoints
    stim = stim[:KEEP_TIMEPOINTS]
    spikes = spikes[:KEEP_TIMEPOINTS]

    # Predict spike counts
    predicted_counts = predict_spike_counts_lg(stim, spikes)

    # Visualize
    plot_spikes_with_prediction(spikes, predicted_counts, dt_stim)

if __name__ == "__main__":
    main()
```

---
