# CS59300CVD Assignment 1: Prokudin-Gorskii Plate Alignment

**Author:** Md Kamruzzaman Kamrul  
**Email:** mkamrul@purdue.edu  
**Course:** CS59300CVD - Computer Vision with Deep Learning (Spring 2026)

---

## 📌 Overview
This project implements an automated system to align digitized Prokudin-Gorskii glass plate photographs. The algorithm takes a single vertical glass plate image, separates it into Blue, Green, and Red channels, and aligns them to produce a high-quality color reconstruction.

The solution features a **coarse-to-fine image pyramid**, a robust **three-stage border removal strategy**, and vectorized operations using PyTorch, reducing runtime from minutes to seconds per image.

---

## 🚀 Key Features

* **⚡ Image Pyramid Search:** Uses a recursive Gaussian pyramid to handle large displacements efficiently ($O(\log W)$ complexity).
* **✂️ Three-Stage Border Removal:**
    1.  **Global:** Adaptive 15th-percentile thresholding to strip jagged glass edges.
    2.  **Per-Channel:** A forced 5% crop on individual channels to remove colored bands that confuse metrics (critical for MSE).
    3.  **Intersection:** Post-alignment cropping to remove "ghosting" artifacts.
* **📐 Multiple Metrics:** Supports **NCC** (Normalized Cross-Correlation), **MSE** (Mean Squared Error), and **SSIM** (Structural Similarity).
* **🏎️ Vectorized Implementation:** Utilizes `torch.roll` and tensor operations instead of slow Python loops.

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.8 or higher
* pip

### 1. Install Dependencies
Run the following command to install the required libraries (PyTorch, NumPy, Pillow, scikit-image):

```bash
pip install -r requirements.txt

### 2. Verify Data
Ensure your input images are located in a `data/` folder in the root directory:

project_root/
├── alignment_model.py      # Core logic (Pyramid, Cropping, Loop)
├── metrics.py              # Metric definitions (NCC, MSE, SSIM)
├── helpers.py              # Helper functions (torch.roll shifting)
├── main.py                 # Entry point (CLI argument parsing)
├── README.md               # Documentation
├── requirements.txt        # Dependencies
└── .gitignore              # Git ignore file

---

## 💻 Usage

### Process All Images (Recommended)
To run the alignment on all 6 images using all 3 metrics (NCC, MSE, SSIM):

```bash
python main.py -i all

This will generate aligned images in the root directory (e.g., 1_ncc_aligned.png, 1_mse_aligned.png, etc.).

### Process a Single Image
To align a specific image using a specific metric:

```bash
python main.py -i data/1.jpg -m ncc

## 🧠 Implementation Details

### 1. The Algorithm Pipeline
The core logic resides in `alignment_model.py`. The process follows a multi-scale approach:
1.  **Pre-processing:** The raw image is split into B/G/R thirds.
2.  **Pyramid Construction:** Images are recursively downsampled by a factor of 2 until the width is < 64 pixels.
3.  **Search:**
    * **Base Level:** Exhaustive search over `[-15, 15]` pixels.
    * **Refinement:** Upscale the best shift by 2, then perform a local search over `[-2, 2]` pixels.
4.  **Metric Evaluation:** Scores are computed on the **Green channel anchor**. Crucially, a **10% border** is temporarily cropped during metric calculation to avoid edge artifacts.

### 2. The "Three-Stage" Border Strategy
Standard cropping was insufficient for high-quality results. We implemented a robust pipeline:
* **Stage 1 (Detection):** We compute row/column intensity averages and use `torch.quantile(..., 0.15)` to detect and crop the dark glass plate borders.
* **Stage 2 (Cleaning):** We apply a hard **5% crop** to the perimeter of each separated channel. *Why?* We found that solid colored lines (e.g., green borders on the blue channel) persisted after Stage 1, causing MSE to fail on the Emir image. This step fixed that failure.
* **Stage 3 (Cleanup):** After alignment, shifting creates zero-padded edges. We calculate the geometric intersection of the valid regions and crop the final output to strictly remove these "dead" pixels.

## 📊 Performance ##

| Feature | Naive Implementation | Our Pyramid Implementation |
| :--- | :--- | :--- |
| **Complexity** | $O(N \cdot W^2)$ | $O(N \cdot \log W)$ |
| **Runtime** | ~45-60 seconds/image | **2-5 seconds/image** |
| **Search Space** | 961 candidates | ~50-100 candidates |

---

## 📂 File Structure

```text
.
├── main.py                 # Entry point (CLI argument parsing)
├── alignment_model.py      # Core logic (Pyramid, Cropping, Loop)
├── metrics.py              # Metric definitions (NCC, MSE, SSIM)
├── helpers.py              # Helper functions (torch.roll shifting)
├── requirements.txt        # Dependencies
├── README.md               # This file
└── data/                   # Input images directory
    ├── 1.jpg
    └── ...

