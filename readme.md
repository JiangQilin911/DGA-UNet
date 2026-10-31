# DGA-UNet: Dual-Branch Gated-Fusion Attention U-Net for Submarine Landslide Segmentation

Official PyTorch implementation of **DGA-UNet**, a dual-branch deep learning framework for multimodal semantic segmentation of submarine landslides using bathymetric and seismic attributes.

This repository provides the model architecture, data preprocessing utilities, training and inference scripts, evaluation tools, and selected cropped GeoTIFF samples for code testing and demonstration.

## 1. Overview

Accurate identification of submarine landslides requires the integration of complementary information from seafloor morphology and subsurface seismic attributes. DGA-UNet is designed to exploit these complementary data sources through a dual-branch feature extraction framework and an adaptive feature-fusion mechanism.

The network takes two co-registered inputs:

- **Attribute Convolution Branch (ACB):** extracts features from three channels, including seafloor slope, curvature, and seismic coherence.
- **Topographic Feature Branch (TFB):** extracts topographic features from four channels, including bathymetric elevation, gradient magnitude, and the sine and cosine components of gradient direction.

The two branches are integrated through a learnable gated-fusion module. The network further incorporates GCBlock and CBAM attention modules to enhance feature representation.

The model is trained using a combination of Dice loss and focal loss. AdamW is used as the optimizer, with a warm-up strategy and cosine annealing learning-rate schedule.

### Key features

- Dual-branch feature extraction from bathymetric and seismic attributes.
- Learnable gated fusion for adaptive multimodal feature integration.
- Attention-enhanced feature representation using GCBlock and CBAM.
- Joint Dice and focal loss for semantic segmentation.
- Training, evaluation, and inference utilities implemented in PyTorch.

## 2. Performance

The DGA-UNet model achieved the following performance on the complete experimental dataset used in the study:

| Metric | Value |
|---|---:|
| Intersection over Union (IoU) | 84.6% |
| F1-score | 91.6% |
| Precision | 92.1% |
| Recall | 91.2% |

**Important:** These metrics were obtained using the complete experimental dataset described in the associated manuscript. The limited GeoTIFF samples released in this repository are provided for software testing and demonstration and are not sufficient to reproduce these reported results.

## 3. Repository Structure

The repository is organized as follows:

```text
DGA-UNet/
│
├── datasets/
│   └── ...
│
├── models/
│   └── ...
│
├── utils/
│   └── ...
│
├── examples/
│   ├── data/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   ├── dems/
│   │   │   └── masks/
│   │   └── val/
│   │       ├── images/
│   │       ├── dems/
│   │       └── masks/
│   │
│   └── ...
│
├── train.py
├── test.py
├── predict.py
├── requirements.txt
├── LICENSE
└── README.md
```

The directory structure above describes the main components of the repository. Please refer to the actual repository contents for the available scripts and modules.

## 4. Environment and Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA 11.8 or a compatible CUDA environment for GPU acceleration
- Other dependencies listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/JiangQilin911/DGA-UNet.git
cd DGA-UNet
```

Create and activate a virtual environment (optional but recommended):

```bash
conda create -n dga_unet python=3.10 -y
conda activate dga_unet
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Install a PyTorch version compatible with your operating system, GPU, and CUDA environment if it is not already installed.

## 5. Data Availability and Preparation

### 5.1 Data availability

The original three-dimensional seismic volume and multibeam bathymetric dataset used in this study are subject to commercial licensing and survey-data restrictions and therefore cannot be publicly redistributed in their entirety.

To facilitate code testing and demonstrate the implementation of DGA-UNet, this repository provides selected cropped GeoTIFF samples derived from the original survey data.

These samples are genuine data crops rather than artificially generated or synthetic images. They represent only a limited subset of the original experimental data. The publicly available training samples include augmented images, while the validation samples are also limited in number.

The released samples are intended to:

- Demonstrate the required input data structure.
- Illustrate the preprocessing and data-loading workflow.
- Facilitate functional testing of the model implementation.
- Provide examples for testing training, evaluation, and inference procedures.

Because the released samples are limited and include augmented training images, they should not be regarded as an independent or representative dataset for assessing model generalization. They are also insufficient to reproduce the full quantitative results reported in the manuscript.

### 5.2 Data organization

The example data are organized into training and validation directories:

```text
examples/data/
│
├── train/
│   ├── images/
│   ├── dems/
│   └── masks/
│
└── val/
    ├── images/
    ├── dems/
    └── masks/
```

Each sample consists of three corresponding GeoTIFF files:

| Directory | Data | Description |
|---|---|---|
| `images/` | Three-band GeoTIFF | Slope, curvature, and seismic coherence |
| `dems/` | Single-band GeoTIFF | Bathymetric elevation |
| `masks/` | Single-band GeoTIFF | Binary submarine-landslide labels |

The three bands in `images/*.tif` are arranged as follows:

- Band 1: Seafloor slope.
- Band 2: Seafloor curvature.
- Band 3: Seismic coherence.

The `dems/*.tif` files contain bathymetric elevation values.

The `masks/*.tif` files contain binary segmentation labels:

- `0`: Background.
- `1`: Submarine landslide.

Corresponding files should have matching filenames and spatial dimensions. Their coordinate reference systems and spatial resolutions should also be consistent.

### 5.3 Input preprocessing

The input data should follow the preprocessing procedures implemented in the repository.

For the ACB branch, slope, curvature, and seismic coherence are represented as three input channels. In the experimental workflow, these attributes are linearly stretched to the range of 0–255.

For the TFB branch, bathymetric elevation and its derived topographic features are used as four input channels:

1. Normalized bathymetric elevation.
2. Gradient magnitude.
3. Sine of gradient direction.
4. Cosine of gradient direction.

The preprocessing procedures should be applied consistently to the training, validation, and inference data.

For users preparing their own datasets, the input files should be spatially co-registered and should follow the same channel definitions and preprocessing conventions as the released examples.

## 6. Running the Code

The repository includes scripts for model training, evaluation, and prediction.

Before running the scripts, check their available command-line arguments:

```bash
python train.py --help
python test.py --help
python predict.py --help
```

The following commands illustrate the main workflow:

### 6.1 Model training

```bash
python train.py
```

The training script loads the configured training and validation data, initializes DGA-UNet, and performs model optimization.

For experiments using custom datasets, configure the data paths and training parameters according to the arguments supported by `train.py`.

### 6.2 Model evaluation

```bash
python test.py
```

The evaluation script evaluates a trained model using the configured test data and reports the corresponding segmentation metrics.

A trained model checkpoint and appropriately formatted evaluation data are required.

### 6.3 Model inference

```bash
python predict.py
```

The prediction script performs semantic segmentation using a trained model checkpoint and the configured input data.

Please refer to the command-line arguments and configuration settings in the corresponding scripts when specifying model weights and input paths.

**Note:** The commands above describe the functions of the included scripts. The actual data paths, checkpoint paths, and optional arguments may need to be configured according to the local environment. Outputs obtained using the limited example samples are intended for software testing and should not be interpreted as reproductions of the full experimental results.

## 7. Custom Dataset

Users can prepare their own datasets by following the directory structure and data-format requirements described in Section 5.

For a new dataset, the following conditions should be satisfied:

1. Bathymetric and seismic-derived attributes should be spatially co-registered.
2. All input channels should follow the definitions and preprocessing procedures expected by the model.
3. Segmentation masks should use consistent binary labels.
4. Corresponding input images, DEMs, and masks should have matching filenames and spatial dimensions.
5. Training and validation data should be separated before data augmentation to avoid data leakage.

The training, evaluation, and prediction scripts can then be configured to use the corresponding dataset paths and model checkpoints.

## 8. Reproducibility and Data Limitations

The source code and selected cropped GeoTIFF samples are publicly available to support inspection and functional testing of the proposed method.

However, the complete original three-dimensional seismic volume and multibeam bathymetric dataset cannot be publicly redistributed because of commercial licensing and survey-data restrictions.

Consequently, the public repository does not contain the complete experimental dataset used to train and evaluate the model in the associated manuscript. The released samples are intended to demonstrate the implementation and input format, rather than to provide a complete benchmark dataset.

The quantitative performance reported in the manuscript was obtained using the complete experimental dataset, not solely the publicly released example samples.

## 9. Citation

If you use this code in your research, please cite the associated manuscript:

> Jiang, Q., Liu, X., Tao, H., et al. Adaptive Multimodal Semantic Segmentation of Submarine Landslides Using Bathymetric and Seismic Attributes. *Computers & Geosciences*. [Publication details to be added after publication.]

## 10. License

This project is distributed under the license specified in the `LICENSE` file.

Please refer to that file for the terms and conditions governing the use, modification, and distribution of the source code.

The license for the source code does not automatically grant permission to redistribute the original survey data or any derived data products. Users should independently verify the applicable data-use restrictions before redistributing or reusing the data.

## 11. Contact

For questions about the model implementation or code, please open an issue in this GitHub repository:

https://github.com/JiangQilin911/DGA-UNet
