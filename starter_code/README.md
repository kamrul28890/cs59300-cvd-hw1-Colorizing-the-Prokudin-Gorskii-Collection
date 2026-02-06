# CS59300CVD Assignment 1: Prokudin-Gorskii Plate Alignment

**Author:** Md Kamruzzaman Kamrul\
**Email:** mkamrul@purdue.edu\
**Course:** CS59300CVD -- Computer Vision with Deep Learning (Spring
2026)

------------------------------------------------------------------------

## 📌 Overview

This project implements an automated system to align digitized
Prokudin-Gorskii glass plate photographs. Each input image contains
three vertically stacked exposures corresponding to the **Blue**,
**Green**, and **Red** channels. The system separates these channels,
aligns them, and reconstructs a high-quality color image.

The implementation uses a **coarse-to-fine image pyramid**, a robust
**three-stage border removal strategy**, and **vectorized PyTorch
operations**, reducing runtime from minutes to seconds per image.

------------------------------------------------------------------------

## 🚀 Key Features

-   **⚡ Image Pyramid Search**\
    A recursive Gaussian pyramid enables efficient alignment for large
    displacements with\
    time complexity O(log W).

-   **✂️ Three-Stage Border Removal**

    1.  **Global Detection:** Adaptive 15th-percentile thresholding
        removes dark glass plate edges.
    2.  **Per-Channel Cleaning:** A forced 5% crop on each channel
        removes residual color bands that confuse metrics (especially
        MSE).
    3.  **Intersection Cleanup:** Post-alignment cropping removes
        zero-padded and ghosting artifacts.

-   **📐 Multiple Similarity Metrics**

    -   NCC (Normalized Cross-Correlation)
    -   MSE (Mean Squared Error)
    -   SSIM (Structural Similarity Index)

-   **🏎️ Vectorized Implementation** Uses torch.roll and tensorized
    operations instead of slow Python loops.

------------------------------------------------------------------------

## 🛠️ Installation & Setup

### Prerequisites

-   Python 3.8 or higher
-   pip

### Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📂 Project Structure

    .
    ├── main.py
    ├── alignment_model.py
    ├── metrics.py
    ├── helpers.py
    ├── requirements.txt
    ├── README.md
    └── data/
        ├── 1.jpg
        └── ...

------------------------------------------------------------------------

## 💻 Usage

### Align All Images

``` bash
python main.py -i all
```

### Align a Single Image

``` bash
python main.py -i data/1.jpg -m ncc
```

Supported metrics: ncc, mse, ssim

------------------------------------------------------------------------

## 📊 Performance

| Feature     | Naive             | Pyramid           |
|------------|-------------------|-------------------|
| Complexity | O(N·W²)           | O(N·log W)        |
| Runtime    | 45–60 s/image     | 2–5 s/image       |
| Candidates | 961               | ~50–100           |

------------------------------------------------------------------------

## 📎 References

-   Prokudin-Gorskii Collection, Library of Congress
-   Szeliski, *Computer Vision: Algorithms and Applications*
