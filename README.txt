## Overview

This repository contains the data, trained models, and code needed to reproduce the main figures from the paper:

"Machine vision with small numbers of detected photons per inference"

Shi-Yuan Ma, Jérémie Laydevant, Mandar M. Sohoni, Logan G. Wright, Tianyu Wang, and Peter L. McMahon

## The PANS Framework

Photon-aware neuromorphic sensing (PANS) is a framework for optimizing sensing systems under extreme photon-budget constraints. PANS models the stochastic single-photon detection process exactly as it physically occurs, rather than relying on conventional surrogate approximations. By faithfully incorporating detection stochasticity into the forward pass and using customized gradient estimation techniques for backpropagation, PANS enables effective end-to-end optimization of both the physical optical front end and the digital back end, jointly optimized through the lossy, stochastic detection bottleneck under stringent physical resource constraints.

## Getting Started

This repository is fully self-contained: all reported results can be reproduced end-to-end from the trained models and raw experimental data, not just replotted from saved numbers. The notebooks in `test/` run the trained PANS models on the provided datasets (and, for FashionMNIST/MNIST, on the experimentally collected photon counts) to regenerate the result files in `results/`. The notebooks in `main_figures/` then produce the plots in the main text (Figs. 2–5) from those results.

### Environment setup

The code was developed and tested with Python 3.13. To set up the environment:

    conda create --name pans python=3.13 pip
    conda activate pans
    pip install -r requirements.txt

Note: MNIST and FashionMNIST datasets will be automatically downloaded by torchvision on first run into `data/MNIST/` and `data/FashionMNIST/`.

## Repository Structure

### data/
Datasets used for testing, including standard MNIST and FashionMNIST (auto-downloaded by torchvision on first run) and simulated datasets for barcode identification, cell classification, transient event detection, and fiber end-face inspection (stored as `.npz` files).

### data/exp_actv/
Experimentally collected single-photon detector (SPD) activation data, including detected photon counts from the OLED + qCMOS hardware setup used in the FashionMNIST and MNIST experiments.

### main_figures/
Jupyter notebooks (Fig2–Fig5) for generating the data plots in the main-text figures, plus `include.py` for shared imports, path definitions, and plotting configuration.

### misc/
- `plot.mplstyle`: Matplotlib style sheet for consistent figure formatting.

### models/
Trained PANS model weights (PyTorch `.pth` files), organized by task into subdirectories. Includes models for both active PANS (FashionMNIST, MNIST, barcode, cell) and passive PANS (MMF classification, MMF reconstruction, transient, fiber, nebula) experiments and simulations.

### plots/
Output directory for generated figure panels (PNG files). These are produced by the notebooks in `main_figures/`.

### requirements.txt
Python package versions used to run the code.

### results/
Precomputed result data (NumPy pickle files) produced by notebooks in `test/` and consumed by notebooks in `main_figures/`.

Key naming convention:
- `*_active_exp.npy`: Experimental results for active PANS tasks.
- `*_active_sim.npy`: Simulation results for active PANS tasks.
- `*_passive_sim.npy`: Simulation results for passive PANS tasks.
- `*_directimg.npy`: Direct-imaging baseline results.
- `*_nonpae2e.npy`: Non-photon-aware end-to-end baseline.
- `*_conf_mats.npz`: Confusion matrices.

### src/
Core Python modules:
- `PANS_linear.py`: Model definitions for PANS with linear optical front ends, including `PANS_active` (structured illumination) and `PANS_passive` (passive optical encoder) classes, with built-in photon-aware stochastic forward propagation and robust testing methods.
- `PhotonActivation.py`: Custom PyTorch activation functions implementing stochastic single-photon detection with Poisson statistics and the damped straight-through estimator (Eq. 1 in the paper).
- `SSIM.py`: Structural similarity index (SSIM) computation for evaluating image reconstruction quality.

### test/
Jupyter notebooks for evaluating trained PANS models and generating result data:
- `test_FashionMNIST.ipynb`: Active PANS evaluation on FashionMNIST (experiment + simulation).
- `test_MNIST.ipynb`: Active PANS evaluation on MNIST (experiment + simulation).
- `test_barcode1010.ipynb`: Active PANS evaluation on barcode identification (simulation).
- `test_cell.ipynb`: Active PANS evaluation on cell classification (simulation).
- `test_passive.ipynb`: Passive PANS evaluation across different tasks: MMF-based MNIST classification, MMF-based MNIST reconstruction, transient event detection, and fiber end-face inspection.
