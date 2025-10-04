Multi-modal Deception Detection on Real-Life Trial Data
This project implements a state-of-the-art multi-modal fusion model for high-stakes deception detection, trained and evaluated on real-life courtroom trial data. It combines linguistic, acoustic, and visual cues to predict truthfulness with high fidelity.

Project Overview
The core objective of this work was to re-implement and significantly enhance a foundational courtroom deception detection model (based on the work of Sen et al.). The system processes multi-modal evidence streams to achieve robust performance in a challenging, real-world context.

The model employs a late-fusion approach, where features extracted from three distinct modalities—text, audio, and video—are independently processed and then combined for a final deception classification decision.

Key Features
Multimodal Fusion: Integrates data from three key modalities:

Text (Linguistic Cues): Processed using the BERT (Bidirectional Encoder Representations from Transformers) model.

Audio (Acoustic Cues): Extracted using MFCCs (Mel-Frequency Cepstral Coefficients).

Video (Visual Cues): Analyzed via Facial Recognition and Computer Vision techniques to capture non-verbal behavior.

Reproducible Baseline: Successfully re-implemented the original Sen et al. model architecture, faithfully reproducing the baseline performance of approximately 84% accuracy under a strict Leave-One-Subject-Out (LOSO) cross-validation split.

Performance Optimization: Through iterative analysis of dataset size versus network depth trade-offs, the final hybrid model achieved a peak accuracy of 94% and an F1 Score of 0.94 on the limited courtroom dataset.

Built on PyTorch: The entire deep learning pipeline is built and managed using the PyTorch framework.

Performance and Results
Metric

Baseline (Sen et al. Reproduction)

Optimized Hybrid Model

Accuracy

≈ 84% (under LOSO)

94%

F1 Score

N/A

0.94

Cross-Validation

Strict Leave-One-Subject-Out (LOSO)

LOSO

Technology Stack
Category

Key Technologies

Deep Learning

PyTorch, Hugging Face Transformers (for BERT)

Linguistic Processing

BERT

Audio/Video Processing

Libraries for MFCC extraction (e.g., Librosa), and Facial Recognition/CV (e.g., OpenCV, Dlib)

Data & Core

Python 3, Pandas, NumPy

Repository Structure
The core implementation is contained within a series of Jupyter Notebooks, detailing the pipeline from data preparation to final evaluation.

File/Directory

Description

Pre-Process.ipynb / pre-process 3.ipynb

Scripts for data cleaning, synchronization, and feature extraction (text, audio, and video modalities).

MDDM 1.ipynb

Notebook focusing on unimodal model training and initial experimentation.

MDDM 2.ipynb

Notebook detailing feature fusion techniques and early hybrid model implementation.

MDDM 3.ipynb

Final notebook containing the optimized multi-modal architecture, training loop, and hyperparameter tuning that achieved 94% accuracy.

deception_evaluation.py

Python script for running standardized evaluation metrics and generating performance reports.

LICENSE

MIT License.

Setup and Usage
Prerequisites
You will need Python 3.x and the necessary deep learning and data science packages.

Note: To fully reproduce this work, you must obtain access to the Real-Life Trial Data (the specific dataset referenced in the Sen et al. paper), as it is not included in this repository due to privacy restrictions.

Installation
Clone the repository:

git clone [https://github.com/Harsh-Pachouri/Multi-Modal-Deception-Detection-on-Real-Life-Trial-Data.git](https://github.com/Harsh-Pachouri/Multi-Modal-Deception-Detection-on-Real-Life-Trial-Data.git)
cd Multi-Modal-Deception-Detection-on-Real-Life-Trial-Data

Action Required: Please provide the contents of the requirements.txt file or a list of major dependencies to complete this installation section.

Running the Project
Data Preparation: Place the pre-processed data files in the expected directory structure (refer to the Pre-Process.ipynb notebook for required file formats and paths).

Feature Engineering: Run the pre-processing notebooks (Pre-Process.ipynb / pre-process 3.ipynb) to extract features for all modalities.

Model Training: Step through the MDDM 1.ipynb to MDDM 3.ipynb notebooks sequentially to train and evaluate the unimodal and final multi-modal fusion models.

Evaluation: Use the deception_evaluation.py script to reproduce the performance metrics.
