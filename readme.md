# GAN-based Gene Expression Prediction

This repository contains Python code for model construction, training, and evaluation
of a GAN-based framework for gene expression prediction across multiple datasets.

## File structure

- `data_utils.py`  
  Data loading and preprocessing functions.

- `models.py`  
  Definition of the generator and discriminator models.

- `metrics.py`  
  Evaluation metrics used in this study.

- `train.py`  
  Training and cross-validation logic.

- `run_experiment.py`  
  Main entry script for running experiments on a given dataset.

## Requirements

- Python >= 3.8
- torch
- numpy
- pandas
- scikit-learn

You can install the required packages using:

```bash
pip install torch numpy pandas scikit-learn
