---
title: Tea Leaf Disease Classifier
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.12.0
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 🍃 Tea Leaf Disease Classifier

A deep learning model based on the **Swin Transformer** architecture for classifying tea leaf diseases from images. 

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/AngryPakhi/tea-leaf-disease-detection)
[![Model](https://img.shields.io/badge/Model%20Weights-Kaggle-blue?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/models/decodedasad/tea-leaf-swin-lowlight)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-green?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/datasets/decodedasad/tea-leaf-diseases-dataset-augmented-splited)

## Project Overview

This repository contains the inference code and application structure for a Gradio web app that classifies 12 categories of tea leaf conditions (9 diseases, 2 pests, and a healthy class). The model architecture used is a `Swin Transformer Tiny` (224×224 input), which has been specifically optimized for low-light conditions to simulate real-world field photography.

**Note:** The model weights (`best_swin.h5`) are not included in this repository due to size constraints. You can download the trained model from the Kaggle link above.

## Key Features

- **Architecture**: Swin Transformer Tiny
- **Classes**: 12 (9 diseases + 2 pests + 1 healthy)
- **Performance**: ~94% Test Accuracy
- **Training Data**: ~30,000 augmented images
- **Framework**: TensorFlow / Keras (Legacy Keras 2 mode)
- **Interface**: Gradio Blocks with real-time prediction visualization

## Supported Classes

| Category | Classes |
|----------|---------|
| **Diseases** | Algal Spot, Anthracnose, Bird Eye Spot, Brown Blight, Gray Blight, Helopeltis, Red Leaf Spot, Red Rust, White Spot |
| **Pests** | Green Mirid Bug, Red Spider |
| **Healthy** | Normal healthy leaves |

## How to Run Locally

If you wish to run this application on your local machine:

1. Clone this repository:
   ```bash
   git clone https://github.com/AsadIslam111/Tea-Leaf-Diseases-Detection.git
   cd Tea-Leaf-Diseases-Detection
   ```

2. Download the model weights (`best_swin.h5`) from the [Kaggle Model link](https://www.kaggle.com/models/decodedasad/tea-leaf-swin-lowlight) and place the file in the root directory of this repository.

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Gradio app:
   ```bash
   python app.py
   ```

## Usage

1. Open the web interface (locally or via the Hugging Face Space).
2. Upload a clear photo of a tea leaf.
3. The model predicts the top-5 most likely conditions.
4. Each prediction includes a confidence percentage visualized as a bar chart.

## Technical Details

- **Preprocessing**: ImageNet normalization (torch mode).
- **Optimizer**: AdamW with Cosine Decay LR schedule during training.
- **Augmentation (Training)**: Random flips, brightness, contrast, saturation, low-light simulation.

## License
MIT License. See the `LICENSE` file for details.
