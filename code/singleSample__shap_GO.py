from binn import BINN, Network
import pandas as pd
import numpy as np
import torch
import random
import os
from docs.util_for_examples import fit_data_matrix_to_network_input
from binn.explainer import BINNExplainer
from binn.based_cell_train import based_cell_train
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle
from datetime import datetime
from collections import defaultdict
import dill


def _generate_k_folds(
        data_matrix: pd.DataFrame,
        design_matrix: pd.DataFrame
):
 
    dfs = []
    sample_names = []  

    group_samples = design_matrix["sample"].values
    df_group = data_matrix[group_samples].T 
    dfs.append(df_group)
    sample_names.extend(group_samples)  
    X = pd.concat(dfs).fillna(0).to_numpy()

    return X,  sample_names 

def _fit_data_matrix_to_network_input(
         data_matrix: pd.DataFrame, features, feature_column="Gene"
    ) -> pd.DataFrame:
        nr_features_in_matrix = len(data_matrix.index)
        if len(features) > nr_features_in_matrix:
            features_df = pd.DataFrame(features, columns=[feature_column])
            data_matrix = data_matrix.merge(features_df, how="right", on=feature_column)
        if len(features) > 0:
            data_matrix.set_index(feature_column, inplace=True)
            data_matrix = data_matrix.loc[features]
        return data_matrix

test_input_data = pd.read_csv("/home/zxl/hdd/cellfate/data/TCGA_primary_tpm_441.csv")
test_input_sign = pd.read_csv("/home/zxl/hdd/cellfate/data/subtype_primary_tpm_441.csv",sep=',')
test_pathways = pd.read_csv("/home/zxl/hdd/lsy/data/BP_5.csv")
test_translation = pd.read_csv("/home/zxl/hdd/lsy/data/BP_Gene_new.csv")
target_key = "Teff_CD8_cell_Tcm_layers"


with open('/home/zxl/hdd/cellfate/data/Gene_and_network1.pkl', 'rb') as file:
    data = pickle.load(file)
gene_list = data["gene_list"]

network = Network(
    input_data=test_input_data,
    pathways=test_pathways,
    mapping=test_translation,
    input_data_column="Gene",
    source_column="source",
    target_column="target"
)


binn = BINN(
    network=network,
    activation = "tanh",
    activation_final = "sigmoid",
    connectivity_matrices_list = data,
    dropout=0.2,
    validate=False,
    device="cuda:1",
    learning_rate=0.001,
) 


input_data = test_input_data
design_matrix = test_input_sign

fitted_input_data = _fit_data_matrix_to_network_input(
                input_data.reset_index(),
                features=data["gene_list"],
                feature_column="Gene"
            )

X, y= _generate_k_folds(
                data_matrix = fitted_input_data, design_matrix=design_matrix
            )
        
background_data = torch.Tensor(X)
y=y

test_data = torch.Tensor(X)
background_data = torch.Tensor(X)
background_data = test_data



def get_connectivity_matrices_list():
    connectivity_matrices_list = model.B_cell_connectivity_matrices[:4]+ model.CD8_cell_Tcm_connectivity_matrices[:4]+ model.CD8_cell_Tem_connectivity_matrices[:4]+ model.CD4_cell_Th1_connectivity_matrices[:4]+model.CD4_cell_Th2_connectivity_matrices[:4]+ model.CD4_cell_Th17_connectivity_matrices[:4]+model.CD4_cell_Tfh_connectivity_matrices[:4]+ model.CD4_cell_Treg_connectivity_matrices[:4]+ model.B_cell_connectivity_matrices[6:10]+model.B_cell_connectivity_matrices[12:16]+ model.B_cell_connectivity_matrices[18:22]+model.B_cell_connectivity_matrices[24:28]+model.B_cell_connectivity_matrices[30:34]+model.CD8_cell_Tcm_connectivity_matrices[6:10]+model.CD8_cell_Tcm_connectivity_matrices[12:16]+  model.CD8_cell_Tcm_connectivity_matrices[18:22]+ model.CD8_cell_Tcm_connectivity_matrices[24:28]+ model.CD8_cell_Tcm_connectivity_matrices[30:34]+model.CD8_cell_Tem_connectivity_matrices[6:10]+  model.CD8_cell_Tem_connectivity_matrices[12:16]+  model.CD8_cell_Tem_connectivity_matrices[18:22]+ model.CD8_cell_Tem_connectivity_matrices[24:28]+ model.CD8_cell_Tem_connectivity_matrices[30:34]+model.CD4_cell_Th1_connectivity_matrices[6:10]+ model.CD4_cell_Th1_connectivity_matrices[12:16]+ model.CD4_cell_Th1_connectivity_matrices[18:22]+ model.CD4_cell_Th1_connectivity_matrices[24:28]+ model.CD4_cell_Th2_connectivity_matrices[6:10]+ model.CD4_cell_Th2_connectivity_matrices[12:16]+ model.CD4_cell_Th2_connectivity_matrices[18:22]+ model.CD4_cell_Th2_connectivity_matrices[24:28]+model.CD4_cell_Th17_connectivity_matrices[6:10]+ model.CD4_cell_Th17_connectivity_matrices[12:16]+ model.CD4_cell_Th17_connectivity_matrices[18:22]+ model.CD4_cell_Th17_connectivity_matrices[24:28]+model.CD4_cell_Tfh_connectivity_matrices[6:10]+ model.CD4_cell_Tfh_connectivity_matrices[12:16]+ model.CD4_cell_Tfh_connectivity_matrices[18:22]+ model.CD4_cell_Tfh_connectivity_matrices[24:28]+model.CD4_cell_Treg_connectivity_matrices[6:10]+ model.CD4_cell_Treg_connectivity_matrices[12:16]+ model.CD4_cell_Treg_connectivity_matrices[18:22]+ model.CD4_cell_Treg_connectivity_matrices[24:28]


    return connectivity_matrices_list

filename="/home/zxl/hdd/cellfate/model4/binn_model0_tpm_0.001_32.pth"


model = torch.load(filename, map_location="cuda:1", weights_only=False)

explainer =BINNExplainer(model)
        
shap_dict = explainer._explain_cell_layer(
            test_data, y, background_data
            )
#with open('/home/zxl/hdd/cellfate/model4/shap_dict.pkl', 'rb') as file:
#    data = pickle.load(file)
#shap_dict = data["shap_dict"]

feature_dict = {
            "name": [],
            "source name": [],
            "target name": [],
      
            "type": [],
            "source layer":[],
            "target layer": [],                  
    }

samples = y
for sample in samples:
    feature_dict[sample] = []  
    


feature_id_mapping = {}
feature_id = 0
feature_id_mapping["root"] = feature_id
for layer_features in shap_dict["features"]:
    for feature in layer_features:
        feature_id += 1
        feature_id_mapping[feature] = feature_id

curr_layer = 0

connectivity_matrices_list = get_connectivity_matrices_list()
feature_id_mapping = {}
feature_id = 0
feature_id_mapping["root"] = feature_id
for layer_features in shap_dict["features"]:
    for feature in layer_features:
        feature_id += 1
        feature_id_mapping[feature] = feature_id

        curr_layer = 0

values_cell = np.asarray(shap_dict["shap_values"][-1])  
values_cell = abs(values_cell)
values_cell_mean = np.mean(values_cell, axis=0)
element = []
first_elements = [vector[0] for vector in shap_dict["features"][:-1]]
index_dict  = defaultdict(list)

for index, item in enumerate(shap_dict["features"][-1]):
    index_dict[item].append(index)
    element.append(item) 
element = set(element)
prefix_indices = {prefix: [] for prefix in element}


for index, item in enumerate(first_elements):
    for prefix in element:
        if item.startswith(prefix):
            prefix_indices[prefix].append(index)

for prefix, indices in prefix_indices.items():
    print(f"Prefix '{prefix}' found at indices: {indices}")

merged_dict = {}

for key in prefix_indices:

    layers = prefix_indices[key]
    values = index_dict.get(key, [])
    
    merged_dict[key] = [layers, values]



merged_dict[target_key]
first = merged_dict[target_key][0]
second = merged_dict[target_key][1]
name = target_key

shap_dict_items = {"features":[], "shap_values":[]}
connectivity_matrices_list_items = []
for idx in first:
    shap_dict_items["shap_values"].append(shap_dict["shap_values"][idx])
    shap_dict_items["features"].append(shap_dict["features"][idx])
    connectivity_matrices_list_items.append(connectivity_matrices_list[idx])
result_second = values_cell_mean[second]  
curr_layer = 0


for sv, features, cm  in zip(
        shap_dict_items["shap_values"], shap_dict_items["features"], connectivity_matrices_list_items
):
    sv = np.asarray(sv)
            
    sv_mean_final =  sv @ result_second   
    for feature in range(sv_mean_final.shape[1]):

        n_classes = sv_mean_final.shape[2]
        modified_string = features[feature].replace(name+'_', '')
        connections = cm[cm.index == modified_string]
        connections = connections.loc[
                :, (connections != 0).any(axis=0)
        ]  
        for target in connections:
            for curr_class in range(n_classes):
                if curr_class != 1:
                    continue 
                feature_dict["name"].append(name)
                feature_dict["source name"].append(features[feature])
                feature_dict["target name"].append(name + "_" + target)
                feature_dict["type"].append(curr_class)
                feature_dict["source layer"].append(curr_layer)
                feature_dict["target layer"].append(curr_layer + 1)
                sample_index = 0
                for sample in samples:
                    
                    feature_dict[sample].append(sv_mean_final[sample_index][feature][curr_class])
                    sample_index = sample_index + 1    
    curr_layer += 1
        
        
df = pd.DataFrame(data=feature_dict)
output_file = '/home/zxl/hdd/cellfate/model2/shap_GO_teff_Tcm_0.csv'
df.to_csv(output_file, index=False)




