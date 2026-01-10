# DICE
[![Current devel version: 0.1.2](https://img.shields.io/badge/devel%20version-0.1.2-blue.svg)](https://github.com/zhaopapers/DICE)
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

### Construction of GODP learning module 
#### Usage for GODP.ipynb
The GODP.ipynb notebook serves as the main entry point for training the Gene Ontology Biological Process(GODP) learning module. 
It includes data loading, network graph construction, model initialization, and cell subprocess interpretability analysis.
#### Network Construction
```
network = Network(
    input_data==input_data,
    pathways==pathways,
    mapping==translation,
    input_data_column="Gene",  # Column name in your data matrix identifying features
    source_column="source",    # Column name in pathway file for source nodes
    target_column="target"     # Column name in pathway file for target nodes
)
```

#### Training & Validation
The training process is handled by the based_cell_train wrapper, which performs Stratified K-Fold cross-validation and SHAP-based feature importance calculation.
To start training, execute the relevant cell in the notebook:
```
binn = BINN(
    network=network,
    activation = "tanh",
    activation_final = "sigmoid",
    n_layers=4,
    dropout=0.2,
    validate=False,
    device="cuda",
    learning_rate=0.001,
)
explainer = BINNExplainer(binn)
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

#### Model Hyperparameters

`-n_layers` is the number of hidden layers (biological hierarchy levels) to use with a default value of 4(only used in GODP); 

`-activation` is the activation function for hidden layers with a default value of "tanh"(only used in GODP); 

`-activation_final` is the activation function for the final classification layer/residual blocks with a default value of "sigmoid"(only used in GODP); 

`-dropout` is the dropout rate to prevent overfitting with a default value of 0.2(only used in GODP); 

`-learning_rate` is the initial learning rate for the Adam optimizer with a default value of 0.001(only used in GODP); 

`-device` is the compute device ("cpu" or "cuda") with a default value of "cuda";

`-nr_iterations` is the number of training repetitions to perform to ensure robustness, with a default value of 10(only used in GODP);

`-max_epochs` is the maximum number of training epochs per fold, with a default value of 100(only used in GODP);

`-batch_size` is the number of samples per batch during training, with a default value of 16(only used in GODP);

`-n_folds` is the number of cross-validation folds to use, with a default value of 3(only used in GODP);

`-num_workers` is the number of CPU subprocesses used for data loading, with a default value of 20(only used in GODP);

`-dir` is the directory path where the model checkpoints (.pth) and interpretation files (.pkl) will be saved(only used in GODP).

#### Outputs
After execution, the notebook generates two primary CSV files and several checkpoints:

1. Feature Importance File `cell_output_file.csv`
Contains the SHAP values explaining the model's decisions.

2. Metrics File `cell_return_dict.csv`
Contains performance metrics for each iteration and fold.

3. Located in the directory specified by the dir parameter in trainer.fit():

`model_inter{iteration}_fold{fold}.pth: `Saved model weights.

`variables{iteration}_fold{fold}.pkl: `Serialized test data used for SHAP explanation.

### Construction of DICE  
#### Usage for main.ipynb
The main.ipynb notebook is designed for construction of DICE, model evaluation and interpretability. It allows you to load pre-trained BINN models and perform iterative SHAP (SHapley Additive exPlanations) analysis to identify key biological drivers in multi-class datasets (e.g., different cell types in scRNA-seq).
```
binn = BINN(
    activation = "tanh",
    activation_final = "sigmoid",
    connectivity_matrices_list = data,
    dropout=0.2,
    validate=False,
    device="cuda:0",
    learning_rate=0.001,
)

trainer = DICE(binn)
return_dict= trainer.fit(test_input_data,
                        test_input_sign,
                        nr_iterations=1,
                        temperature_init=1.0,
                        connectivity_matrices_list = data,
                        batch_size=32,
                        n_folds=3,
                        val_size = 0.2,
                        test_size = 0.2,
                        max_epochs=100,
                        num_workers=0,
                        gene_list=gene_list)
```
#### Model Hyperparameters

`-activation` is the activation function for hidden layers with a default value of "tanh"(only used in DICE); 

`-activation_final` is the activation function for the final classification layer/residual blocks with a default value of "sigmoid"(only used in DICE); 

`-dropout` is the dropout rate to prevent overfitting with a default value of 0.2(only used in DICE); 

`-learning_rate` is the initial learning rate for the Adam optimizer with a default value of 0.001(only used in DICE); 

`-device` is the compute device ("cpu" or "cuda") with a default value of "cuda"(only used in DICE);

`-nr_iterations` is the number of complete training cycles to run with a default value of 1(only used in DICE);

`-max_epochs` is the maximum number of training epochs per fold with a default value of 100(only used in DICE);

`-batch_size` is the number of samples processed before the model is updated with a default value of 32(only used in DICE);

`-n_folds` is the number of folds for Stratified Shuffle Split cross-validation with a default value of 3(only used in DICE);

`-val_size` is the proportion of data reserved for validation with a default value of 0.2(only used in DICE);

`-test_size` is the proportion of data reserved for testing with a default value of 0.2(only used in DICE);

`-temperature_init` is the initial value for temperature scaling used in probability calibration with a default value of 1.0(only used in DICE);

`-num_workers` is the number of CPU subprocesses to use for data loading with a default value of 0(only used in DICE);

`-connectivity_matrices_list` is the dictionary containing sparse matrices that define the biological network topology(only used in DICE);

`-gene_list` is the list of genes used to filter the input data to ensure intersection with the network features(only used in DICE).

#### Model and Output configuration

model_path = '/path/to/your/binn_model_tpm.pth'
output_dir = '/path/to/results/'

### Model Interpretation
#### Usage for explain.ipynb
The explain.ipynb notebook implements model interpretability analysis by calculating SHAP (SHapley Additive exPlanations) values to quantify feature importance.
```
Calculate the MEAN SHAP values across the group (Group-level resolution).
shap = SHAPExplainer(
            input_data=test_input_data, 
            design_matrix=test_input_sign, 
            model=model,
            device="cuda:0"
        )

shap.explain_cell(
            output_dir="/model", 
            iteration=0

        )

shap = SHAPExplainer(
            input_data=test_input_data, 
            design_matrix=test_input_sign, 
            model=model,
            device="cuda:1"
        )

shap.explain(
            output_dir="/model", 
            iteration=0
   
        )
```


```
Calculate SHAP values for EACH individual sample (Sample-level resolution).
shap_dict = explainer._explain_cell_layer(
            test_data, y, background_data
            )
df_GO = explainer.shap_single_G0(shap_dict,y,connectivity_matrices_list,target_key)
df_cell = explainer.shap_single_cell(shap_dict,y)  


```
#### Model Hyperparameters

`-model` is the trained BINN model object loaded from a saved checkpoint (e.g., .pth file);

`-output_dir` is the directory path where the resulting SHAP explanation CSV files (e.g., shap_cell_iter0.csv) will be saved;

`-iteration` is an integer index used to suffix the output filenames to track different explanation runs with a default value of 0;

`-target_key` is the specific pathway or layer name to focus on when generating single Gene Ontology (GO) term explanations (e.g., "Teff_CD8_cell_Tcm_layers");

`-background_data` is the reference dataset (tensor) derived from training data, used by the SHAP DeepExplainer as the background distribution;

`-test_data` is the target dataset (tensor) for which SHAP values are calculated to explain the model's predictions. 

#### Outputs
`shap_explain_singleSample_iterX.csv` will be created in the output directory.Sample-specific SHAP values for the GO layer, providing feature importance scores for each individual sample.

`shap_cell_singleSample_iterX.csv` will be created in the output directory.Sample-specific SHAP values for the Cell layer, providing feature importance scores for each individual sample.

`shap_explain_iterX.csv` will be created in the output directory.Mean SHAP values for the GO layer, representing the average feature importance across the analyzed group.

`shap_cell_iterX.csv` will be created in the output directory.Mean SHAP values for the Cell layer, representing the average feature importance across the analyzed group.

### Files
| File                              | Description                                                                   |
|------------------------------------|------------------------------------------------------------------------|
| GODP/binn.py                             | Defines the core architecture of the BINN model based on PyTorch Lightning, responsible for dynamically constructing hierarchical neural networks according to biological pathway topologies.                            |
| GODP/network.py                          | Responsible for building biological directed graphs and converting the mapping relationships between pathways and genes into connection matrices required between layers of the neural network. |
| GODP/based_cell_train.py                            | Encapsulates the complete training workflow, responsible for executing stratified K-fold cross-validation, model fitting, and calling SHAP for interpretability analysis.                              |
| GODP/util_for_examples.py  | Provides auxiliary tools for data preprocessing, mainly used to align gene expression matrices with network input features and generate standardized training data.                                       |
| main.ipynb                             | it handles loading pre-trained BINN models and performing iterative SHAP analysis on large-scale scRNA-seq datasets.                            |
| binn.py                          | Implements the Biologically Informed Neural Network (BINN) using PyTorch Lightning, creating a hierarchical architecture based on biological pathway topology. |
| DICE.py                            | Implements "Diverse Counterfactual Explanations" logic to improve model robustness and provide alternative explanations for cell fate decisions.                              |
| explain.py  | Contains the SHAPExplainer class, specialized in calculating and aggregating SHAP values for multi-class cellular data.                                       |
| explainer.py  | Provides the base BINNExplainer logic, including weight initialization and core routines for computing feature importance across hierarchical layers.                                       |
| network.py  | A graph-theory utility that converts biological pathway relations into connectivity matrices and masks for the neural network layers.                                       |

### Data
`test_input_data:` Gene Expression Matrix, which here refers to single-cell data with genes as rows and cells as columns.Supports both single-cell RNA-seq data (e.g., HSC, B-cell, or T-cell differentiation stages) and bulk tissue data (e.g., TCGA Pan-cancer cohorts).

`test_input_sign:` A CSV file providing the metadata for each sample or cell.Must include a sample column (matching test_input_data columns) and a group column (numeric labels, e.g., 1–24) representing ground-truth classes or cell types.

`Pre-trained Models:` Saved .pth files containing the trained BINN architecture and weights.

`Connectivity Maps:` A serialized .pkl file (e.g., Gene_and_network.pkl) containing connectivity matrices for different cell types.

`Benchmark.csv:`A comprehensive log of model performance across iterations, including metrics such as train_acc, test_f1, macro_precision, and specific dataset accuracies (e.g., ICGC, TCGA).

`Supplementary Table.xlsx`The datasets used in this study cover fundamental immune cell differentiation processes and a comprehensive pan-cancer landscape.


