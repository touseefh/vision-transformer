# Vision Transformer Image Classification Pipeline
## Overview
This project is all about using a Vision Transformer (ViT) to classify images into six plant-related classes—like Aphids, Healthy, and Target spot—on a custom dataset. I fine-tuned a pretrained vit_b_16 with PyTorch, setting up a pipeline for data prep, training, and evaluation, with some cool extras like ROC curves and single-image predictions.

## Features
### Data Handling:
Loads a 6-class dataset with train/test splits and applies ViT-specific transforms.
### Model:
Fine-tunes a pretrained ViT, freezing base layers and adding a custom head for 6 classes.
### Training: 
Runs with Cross-Entropy Loss and Adam optimizer, tweaking just the head over 10 epochs.
### Evaluation:
Checks performance with accuracy, confusion matrices, classification reports, and ROC curves.
### Requirements
1. Python 3.x
2. Libraries: PyTorch, Torchvision, Torchaudio, Matplotlib, Scikit-learn, Torchinfo, Pillow
