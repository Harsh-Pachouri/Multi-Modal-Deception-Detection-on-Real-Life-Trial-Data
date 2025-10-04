# Multi-Modal Deception Detection on Real-Life Trial Data

This project implements a state-of-the-art multi-modal fusion model for high-stakes deception detection, trained and evaluated on real-life courtroom trial data. It combines linguistic, acoustic, and visual cues to predict truthfulness with high fidelity.

## Project Overview
The core objective of this research was to not only re-implement a foundational courtroom deception detection model (Sen et al.) but also to significantly enhance its performance and rigorously test its real-world viability and practical feasibility. This involved developing an optimized hybrid architecture and analyzing the trade-offs inherent in using limited, high-stakes judicial data.
The model employs a late-fusion approach, where features extracted from three distinct modalities—text, audio, and video—are independently processed and then combined for a final deception classification decision.

## Key Features
*   **Multi-modal Fusion and Analysis**: Integrates data from three key modalities to capture a holistic view of human behavior, crucial for real-world reliability:
    *   **Text (Linguistic Cues)**: Processed using the BERT (Bidirectional Encoder Representations from Transformers) model.
    *   **Audio (Acoustic Cues)**: Extracted using MFCCs (Mel-Frequency Cepstral Coefficients).
    *   **Video (Visual Cues)**: Analyzed via Facial Recognition and Computer Vision techniques to capture non-verbal behavior.
*   **Reproducible Baseline**: Successfully re-implemented the original Sen et al. model architecture, faithfully reproducing the baseline performance of approximately 84% accuracy under a strict Leave-One-Subject-Out (LOSO) cross-validation split.
*   **Performance Analysis & Enhancement**: Achieved significant performance gains (up to 94% accuracy and 0.94 F1 Score) by iteratively investigating the dataset size vs. network depth tradeoff. This analysis directly informed the model's design for optimal real-world feasibility when deployed with limited, high-value data.
*   **PyTorch Implementation**: The entire deep learning pipeline is built and managed using the PyTorch framework.

## Performance and Results
The enhancement efforts focused on creating a model optimized for reliability and maximizing utility given the scarcity of high-stakes, real-world data.

| Metric | Accuracy |
| --- | --- |
| **Baseline (Sen et al. Reproduction)** | ≈ 84% (under LOSO) |
| **Optimized Hybrid Model** | 94% (under LOSO) |

## Technology Stack

| Category | Key Technologies |
| --- | --- |
| **Deep Learning** | PyTorch, Hugging Face Transformers (for BERT) |
| **Linguistic Processing** | BERT |
| **Audio/Video Processing** | Libraries for MFCC extraction (e.g., Librosa), and Facial Recognition/CV (e.g., OpenCV, Dlib) |
| **Data & Core** | Python 3, Pandas, NumPy |

## Repository Structure
The core implementation is contained within a series of Jupyter Notebooks, detailing the pipeline from data preparation to final evaluation.

| File/Directory | Description |
| --- | --- |
| `Pre-Process.ipynb` / `pre-process 3.ipynb` | Scripts for data cleaning, synchronization, and feature extraction (text, audio, and video modalities). |
| `MDDM 1.ipynb` | Notebook re-implementing the original model |
| `MDDM 2.ipynb` | Notebook detailing feature fusion techniques and early hybrid model implementation. |
| `MDDM 3.ipynb` | Final notebook containing the optimized multi-modal architecture, training loop, and hyperparameter tuning that achieved 94% accuracy. |
| `deception_evaluation.py` | Python script for running standardized evaluation metrics and generating performance reports. |
| `LICENSE` | MIT License. |

## Setup and Usage
### Prerequisites
You will need Python 3.x and the necessary deep learning and data science packages.

**Note**: To fully reproduce this work, you must obtain access to the Real-Life Trial Data (the specific dataset referenced in the Sen et al. paper), as it is not included in this repository due to privacy restrictions.

### Installation
1.  Clone the repository:
    ```sh
    git clone https://github.com/Harsh-Pachouri/Multi-Modal-Deception-Detection-on-Real-Life-Trial-Data.git
    cd Multi-Modal-Deception-Detection-on-Real-Life-Trial-Data
    ```
2.  Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```
    **Note**: The `requirements.txt` file is not provided here. Please ensure it includes dependencies such as `torch`, `transformers`, `librosa`, `opencv-python`, `dlib`, `pandas`, `numpy`, and any other packages required for the project. Refer to the repository for the complete list.

### Running the Project
1.  **Data Preparation**: Place the pre-processed data files in the expected directory structure (refer to the `Pre-Process.ipynb` notebook for required file formats and paths).
2.  **Feature Engineering**: Run the pre-processing notebooks (`Pre-Process.ipynb` / `pre-process 3.ipynb`) to extract features for all modalities.
3.  **Model Training**: Step through the `MDDM 1.ipynb` to `MDDM 3.ipynb` notebooks sequentially to train and evaluate the unimodal and final multi-modal fusion models.
4.  **Evaluation**: Use the `deception_evaluation.py` script to reproduce the performance metrics.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
