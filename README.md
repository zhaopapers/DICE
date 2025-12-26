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

### Usage for base_cell.ipynb
The base_cell.ipynb notebook serves as the main entry point for training the Biologically Informed Neural Network (BINN). 

It orchestrates data loading, network graph construction, model initialization, and the training loop with interpretability analysis.

#### Data

Before running the notebook, ensure you have prepared the four required CSV files.

`test_input_data:`Gene Expression Matrix,which here refers to single-cell data with genes as rows and cells as columns.

`test_input_sign:`Design Matrix.Must contain:- sample: Matches column names in the expression matrix.- group: Class labels (e.g., 1 and 2) used for stratification.

`test_pathways:`Defines the edges of the biological graph.

`test_translation:`Input Mapping.

#### Network Construction

``` Python (version 3.10.9)
``` Anaconda
network = Network(
    input_data=test_input_data,
    pathways=test_pathways,
    mapping=test_translation,
    input_data_column="Gene",  # Column name in your data matrix identifying features
    source_column="source",    # Column name in pathway file for source nodes
    target_column="target"     # Column name in pathway file for target nodes
)
```

| File                              | Description                                                                   |
|------------------------------------|------------------------------------------------------------------------|
| binn.py                             | Defines the core architecture of the BINN model based on PyTorch Lightning, responsible for dynamically constructing hierarchical neural networks according to biological pathway topologies.                            |
| network.py                          | Responsible for building biological directed graphs and converting the mapping relationships between pathways and genes into connection matrices required between layers of the neural network. |
| based_cell_train.py                            | Encapsulates the complete training workflow, responsible for executing stratified K-fold cross-validation, model fitting, and calling SHAP for interpretability analysis.                              |
| util_for_examples.py  | Provides auxiliary tools for data preprocessing, mainly used to align gene expression matrices with network input features and generate standardized training data.                                       |

#### Model Hyperparameters

`-n_layers` is the number of hidden layers (biological hierarchy levels) to use with a default value of 4; 
`-activation` is the activation function for hidden layers with a default value of "tanh"; 
`-activation_final` is the activation function for the final classification layer/residual blocks with a default value of "sigmoid"; 
`-dropout` is the dropout rate to prevent overfitting with a default value of 0.2; 
`-learning_rate` is the initial learning rate for the Adam optimizer with a default value of 0.001; 
`-device` is the compute device ("cpu" or "cuda") with a default value of "cuda".

#### Training & Validation

The training process is handled by the based_cell_train wrapper, which performs Stratified K-Fold cross-validation and SHAP-based feature importance calculation.
To start training, execute the relevant cell in the notebook:

``` Python (version 3.10.9)
``` Anaconda
trainer = based_cell_train(binn, explainer)
df, return_dict = trainer.fit(
    input_data=test_input_data,
    design_matrix=test_input_sign,
    nr_iterations=10,      # Number of total training iterations
    n_folds=3,             # Number of folds for Cross-Validation
    max_epochs=100,        # Maximum training epochs per fold
    batch_size=16,         # Batch size
    num_workers=20,        # Number of CPU workers for data loading
    dir="./models/"        # Directory to save model checkpoints (.pth) and variables
)
```

#### Outputs
After execution, the notebook generates two primary CSV files and several checkpoints:

1. Feature Importance File `proB_preB_output_file_GSE160927.csv`
Contains the SHAP values explaining the model's decisions.

2. Metrics File `proB_preB_return_dict_GSE160927.csv`
Contains performance metrics for each iteration and fold.

3. Located in the directory specified by the dir parameter in trainer.fit():

`model_inter{iteration}_fold{fold}.pth: `Saved model weights.

`variables{iteration}_fold{fold}.pkl: `Serialized test data used for SHAP explanation.

### Usage for main.ipynb
The main.ipynb notebook is designed for high-throughput model evaluation and interpretability. It allows you to load pre-trained BINN models and perform iterative SHAP (SHapley Additive exPlanations) analysis to identify key biological drivers in multi-class datasets (e.g., different cell types in scRNA-seq).

#### Data

`test_input_data:` A CSV file where rows are features (Genes) and columns are samples.

`test_input_sign:` A CSV mapping samples to their ground-truth labels (e.g., classes 1–24).

`Pre-trained Models:` Saved .pth files containing the trained BINN architecture and weights.

`Connectivity Maps:` A serialized .pkl file (e.g., Gene_and_network.pkl) containing connectivity matrices for different cell types.

| File                              | Description                                                                   |
|------------------------------------|------------------------------------------------------------------------|
| main.ipynb                             | it handles loading pre-trained BINN models and performing iterative SHAP analysis on large-scale scRNA-seq datasets.                            |
| binn.py                          | Implements the Biologically Informed Neural Network (BINN) using PyTorch Lightning, creating a hierarchical architecture based on biological pathway topology. |
| DICE.py                            | Implements "Diverse Counterfactual Explanations" logic to improve model robustness and provide alternative explanations for cell fate decisions.                              |
| explain.py  | Contains the SHAPExplainer class, specialized in calculating and aggregating SHAP values for multi-class cellular data.                                       |
| explainer.py  | Provides the base BINNExplainer logic, including weight initialization and core routines for computing feature importance across hierarchical layers.                                       |
| network.py  | A graph-theory utility that converts biological pathway relations into connectivity matrices and masks for the neural network layers.                                       |
| plot.py  | A visualization library for generating publication-quality figures, such as Sankey diagrams, to trace biological feature importance through the network.                                       |

#### Configuration & Initialization
The notebook utilizes the SHAPExplainer to handle the complexity of multi-class biological networks.
``` Python (version 3.10.9)
``` Anaconda
# Model and Output configuration
model_path = '/path/to/your/binn_model_tpm.pth'
output_dir = '/path/to/results/'

shap = SHAPExplainer(
            input_data=test_input_data, 
            design_matrix=test_input_sign, 
            model=model,
            device="cuda:1"
        )
##We systematically identify key biological features (critical genes, functional modules, and their interactions) across different differentiation stages via model interpretation approaches.
shap.explain(
            output_dir="/model", 
            iteration=0
   
        )
##This method can systematically quantify the key biological features that drive cancer classification across scales (cell differentiation subprocesses, biological functional modules, and regulatory genes).
shap.explain_cell(
            output_dir="/model", 
            iteration=0

        )
```
#### Outputs
`shap_values_iteration_{i}.csv` will be created in the output directory.Feature importance scores mapped to biological pathways for each predicted class.




## Citation
Those codes and the CITMIC package are intended for research use only. 

If you use CITMIC or these codes in your publication, please cite the paper: 

X. Zhao, J. Wu, J. Lai, B. Pan, M. Ji, X. Li, Y. He, J. Han, CITMIC: Comprehensive Estimation of Cell Infiltration in Tumor Microenvironment based on Individualized Intercellular Crosstalk. Adv. Sci. 2025, 12, 2408007. https://doi.org/10.1002/advs.202408007
