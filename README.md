## Predicting Pure Fluid Proteries using Transfer Learning ##

This is a FFN that can be trained to predict the dynamic viscosity and thermal conductivity from scratch or using transfer learning. 

## Getting Started

### Installation

Set up conda environment and clone the github repo. The code has been tested on Ubuntu 24.04


- Create a new environment:
```
conda create -n tlearning python=3.11
```
```
conda activate tlearning
```
- Clone the source code of FluidNet-TL:
```
git clone https://github.com/HariOmChadha/FluidNet-TL.git
```
```
cd FluidNet-TL
```
- Install dependencies:
```
pip install torch
pip install -r requirements.txt
```

### Dataset

The datasets used can be found in `data` folder. The splits are done using a specific set of indices to keep consistency. This can be changed in `main.ipynb`.
`cond_254_data.csv`: contains the thermal conductivity values at 5 different temperatures for each molecule and 254 mordred discriptors with low-correlation.
`visc_data.csv`: contains the dynamic viscosity values at 5 different temperatures for each molecule and 254 mordred discriptors with low-correlation.

### Training 

The model can be trained from scratch for either task or a pretrained model can be loaded in to be finetunes on a certain task. The hyperparameters can be adjusted using `config_finetune.yaml`. The models can be trained on different sized train sizes as well.

- Use the jupyter notebook called `main.ipynb` 

### Trained models

We also provide trained FFN models, which can be found in `results`.
- `results/visc/FFN_scrath`: FFN trained using random weights on the dynamic viscosity data
- `results/cond/FF`: FFNs trained using different train set sizes, both using random weights and weights transfered from the viscosity FFN.


