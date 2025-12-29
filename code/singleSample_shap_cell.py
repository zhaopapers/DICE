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
def SHAP_T(dir_data="/home/zxl/hdd/cellfate/data/TCGA_primary_tpm_100.csv",
           dir_sign="/home/zxl/hdd/cellfate/data/subtype_primary_tpm_100.csv",
           dir="/home/zxl/hdd/cellfate/model2/"
           
           ):
    test_input_data = pd.read_csv(dir_data)
    test_input_sign = pd.read_csv(dir_sign,sep=',')
    test_pathways = pd.read_csv("/home/zxl/hdd/lsy/data/BP_5.csv")
    test_translation = pd.read_csv("/home/zxl/hdd/lsy/data/BP_Gene_new.csv")
    with open('/home/zxl/hdd/cellfate/data/Gene_and_network1.pkl', 'rb') as file:
        data = pickle.load(file)
    gene_list = data["gene_list"]

    gene_set = set(gene_list.tolist()) 
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

    X, y = _generate_k_folds(
                    fitted_input_data, design_matrix=design_matrix)

    test_data = torch.Tensor(X)
    background_data = torch.Tensor(X)
    background_data = test_data
    test_data = test_data
    y=y
    for iteration in range(10):
        filename="/home/zxl/hdd/cellfate/model4/binn_model{}_tpm_0.001_32.pth".format(iteration)

       
        model = torch.load(filename, map_location="cuda:1", weights_only=False)
        explainer =BINNExplainer(model)
        
        shap_dict = explainer._explain_cell_layer(
                            test_data, y, background_data
            )

        feature_dict = {
                    "name": [],
       
                    "type": [],
             
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

        for sv, features in zip(
                            shap_dict["shap_values"], shap_dict["features"]
        ):
            sv = np.asarray(sv)
                    


             
            for feature in range(sv.shape[1]):
                n_classes = sv.shape[2]
                for curr_class in range(n_classes):

                    feature_dict["name"].append(features[feature])

                    feature_dict["type"].append(curr_class)             
                        
                    sample_index = 0
                    for sample in samples:
                            
                        feature_dict[sample].append(sv[sample_index][feature][curr_class])
                        sample_index = sample_index + 1
                          

            curr_layer += 1
            print(curr_layer)
        df = pd.DataFrame(data=feature_dict)
        output_file = '{}shap_TCGA_{}.csv'.format(dir,iteration)
        df.to_csv(output_file, index=False)
        print(iteration)



SHAP_T( dir_data="/home/zxl/hdd/cellfate/data/TCGA_primary_tpm_441.csv",
        dir_sign="/home/zxl/hdd/cellfate/data/subtype_primary_tpm_441.csv",
        dir="/home/zxl/hdd/cellfate/model2/"
        )


