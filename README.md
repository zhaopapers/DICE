# DICE
[![Current devel version: 0.1.2](https://img.shields.io/badge/devel%20version-0.1.2-blue.svg)](https://github.com/zhaopapers/CITMIC)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/2ba2ad32650d469588a16de5ae2a5ed1)](https://app.codacy.com/gh/zhaopapers/CITMIC/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

DICE: Decoding Immune Cell Differentiation for Tumor-Type Prediction and Mechanism Interpretation

## Introduction

`DICE` can combines prior knowledge of immune cell differentiation with biological processes to construct a biologically neural network model for tumor type prediction. This is a novel approach  employed feature attribution techniques to conduct an introspective analysis of the model, thereby identifying key immune cell types, biological processes, and genes within differentiation processes that distinguish different cancer types. In conclusion, DICE has been demonstrated to provide a novel tool for clinical pathological diagnosis, offering new insights into the mechanisms of immune cell differentiation within the tumor microenvironment.

![A simple schema of the DICE](Figure/Figure.jpg)

## A notice on operating system compatibility
- We recommend using as input the gene expression matrix normalized by log2(fpkm+0.001).


## Environment set up
installing Anaconda, you can create a virtual environment named DICE and install the required packages based on the environment.yml file using the following command:
``` Python (version 3.10.9)
``` Anaconda
conda env create -f environment.yml
```

## Usage


## Citation
Those codes and the CITMIC package are intended for research use only. 

If you use CITMIC or these codes in your publication, please cite the paper: 

X. Zhao, J. Wu, J. Lai, B. Pan, M. Ji, X. Li, Y. He, J. Han, CITMIC: Comprehensive Estimation of Cell Infiltration in Tumor Microenvironment based on Individualized Intercellular Crosstalk. Adv. Sci. 2025, 12, 2408007. https://doi.org/10.1002/advs.202408007
