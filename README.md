# 3bp-flexing-nn

## Nontrivial Bending in Mechanical Metamaterials Enabled by Strain-Curvature Coupling

This repository contains the ML part (dataset, surrogate model and design search) of the paper

*Nontrivial Bending in Mechanical Metamaterials Enabled by Strain-Curvature Coupling*

by J. Liu, S. Zorkaltsev, L. Wang, D. Pasini, M. Haranczyk and J. Christensen (under review).

A fully connected neural network is trained on finite element (FE) simulations of meta-beams under three-point
bending to predict the end deflection $v_e$ from the unit-cell geometry, the support position $s$ and the
compressive displacement $D$. The trained network is then used as a surrogate to screen new designs with large
negative end deflection.

### Installation

```
pip install -r requirements.txt
```
Tested with Python 3.14, PyTorch 2.14 (CPU), scikit-learn 1.9 and cma 4.5. GPU is not required.

## Dataset

`datasets/fe_dataset_training.csv` contains FE simulations results. Each row is one simulation:

| column | meaning | paper notation | range in dataset |
|---|---|---|---|
| `x1` | x-coordinate of the hinge between the top and bottom quadrilaterals | $x_1/L$ | 0.03 – 0.97 |
| `y3` | y-coordinate of the top hinge at the unit-cell centre | $y_4/L$ | −1.79 – 1.84 |
| `y4` | y-coordinate of the bottom hinge at the unit-cell centre | $y_5/L$ | −2.00 – 1.71 |
| `y7` | y-coordinate of the top hinge at the unit-cell edge | $y_2/L$ | −1.90 – 2.00 |
| `y8` | y-coordinate of the bottom hinge at the unit-cell edge | $y_3/L$ | −2.00 – 1.89 |
| `w` | hinge thickness | $w/2L$ | 0.013 – 0.05 |
| `s` | half span between the pinned supports | $s/2L$ | 0, 0.5, …, 3.5 |
| `v0(V=D)` | end deflection $v_e$ (mm) at compressive displacement $D$ = 0.2, 0.4, …, 10 mm | $v_e$ | −112.8 – 109.8 |


The design parameters were sampled randomly inside the bounds, according to the geometric validity rules
in `threebp/geometry.py` (non-self-intersecting, non-overlapping quadrilaterals, etc). All designs satisfy these rules.

## Model training

The surrogate (`threebp/model.py`) takes 8 inputs `[x1, y3, y4, y7, y8, w, s, D]`, standardized with a
`StandardScaler` fitted on the training subset and returns $v_e$ in mm. It has four hidden layers of
512, 256, 128 and 64 neurons, each followed by batch normalization, ReLU and dropout (p = 0.2). It is trained
with the MSE loss and the Adam optimizer (learning rate 7.5e-5, weight decay 5e-5) for 250 epochs, with a batch
size of 256.

The data points are split randomly into training (80%) and test (20%) subsets. The test loss is monitored during
training, but no model selection or early stopping is applied.

```
python train.py --outdir models/surrogate
```
Training takes about 10 minutes on a CPU. The trained model is provided in `models/surrogate`. The folder contains the weights (`model.pt`), the fitted scaler
(`scaler.joblib`), training configuration and loss history.

## Evaluation

```
python evaluate.py --model models/surrogate
```
This computes metrics on the training and test subsets. It also writes
the prediction for every point (`results/predictions.csv`) and the predicted-vs-FE plot of the test
subset.

## Design search

```
python search.py --model models/surrogate --budget 20000 --seeds 10
```
The surrogate is used to search for designs with the most negative end deflection. Candidates are drawn within the sampling bounds of
the dataset and invalid geometries are rejected before evaluation. Two strategies are compared
with the same number of evaluations.

- **Random screening**: uniform sampling of the design space, as in the paper.
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy, invalid candidates are resampled.

Each strategy is run with several random seeds.

## Repository structure

```
datasets/fe_dataset_training.csv   FE dataset
threebp/data.py                    loading and train/test split
threebp/model.py                   network definition and inference wrapper
threebp/geometry.py                geometric validity rules and sampling bounds
threebp/metrics.py                 regression metrics
train.py                           model training
evaluate.py                        metrics and parity plots
search.py                          random screening vs. CMA-ES on the surrogate
models/surrogate/                  trained model
```
