# DICE
[![Current devel version: 0.1.2](https://img.shields.io/badge/devel%20version-0.1.2-blue.svg)](https://github.com/zhaopapers/CITMIC)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2ba2ad32650d469588a16de5ae2a5ed1)](https://app.codacy.com/gh/zhaopapers/CITMIC/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

DICE: Decoding Immune Cell Differentiation for Tumor-Type Prediction and Mechanism Interpretation

## Introduction

`DICE` can combines prior knowledge of immune cell differentiation with biological processes to construct a biologically neural network model for tumor type prediction. This is a novel approach  employed feature attribution techniques to conduct an introspective analysis of the model, thereby identifying key immune cell types, biological processes, and genes within differentiation processes that distinguish different cancer types. In conclusion, DICE has been demonstrated to provide a novel tool for clinical pathological diagnosis, offering new insights into the mechanisms of immune cell differentiation within the tumor microenvironment.

![A simple schema of the DICE](Figure/figure.jpg)

## A notice on operating system compatibility
- We recommend using as input the gene expression matrix normalized by log2(fpkm+0.001).


## Environment set up
installing Anaconda, you can create a virtual environment named DICE and install the required packages based on the environment.yml file using the following command:
``` Python (version 3.10.9)
``` Anaconda
conda env create -f environment.yml
```

## Usage
#Usage for base_cell.ipynb
The base_cell.ipynb notebook serves as the main entry point for training the Biologically Informed Neural Network (BINN). It orchestrates data loading, network graph construction, model initialization, and the training loop with interpretability analysis.
Data
Before running the notebook, ensure you have prepared the four required CSV files.
test_input_data:Gene Expression Matrix,which here refers to single-cell data with genes as rows and cells as columns.
test_input_sign:Design Matrix.Must contain:- sample: Matches column names in the expression matrix.- group: Class labels (e.g., 1 and 2) used for stratification.
test_pathways:Defines the edges of the biological graph.
test_translation:Input Mapping.
Network Construction
network = Network(
    input_data=test_input_data,
    pathways=test_pathways,
    mapping=test_translation,
    input_data_column="Gene",  # Column name in your data matrix identifying features
    source_column="source",    # Column name in pathway file for source nodes
    target_column="target"     # Column name in pathway file for target nodes
)

Parameter,Default,Description
-n_layers,4,Number of hidden layers (biological hierarchy levels) to use.
-activation,"""tanh""",Activation function for hidden layers.
-activation_final,"""sigmoid""",Activation function for the final classification layer/residual blocks.
-dropout,0.2,Dropout rate to prevent overfitting.
-learning_rate,0.001,Initial learning rate for the Adam optimizer.
-device,"""cuda""","Compute device (""cpu"" or ""cuda"")."




## Citation
Those codes and the CITMIC package are intended for research use only. 

If you use CITMIC or these codes in your publication, please cite the paper: 

X. Zhao, J. Wu, J. Lai, B. Pan, M. Ji, X. Li, Y. He, J. Han, CITMIC: Comprehensive Estimation of Cell Infiltration in Tumor Microenvironment based on Individualized Intercellular Crosstalk. Adv. Sci. 2025, 12, 2408007. https://doi.org/10.1002/advs.202408007
