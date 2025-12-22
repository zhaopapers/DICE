# explain.py
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from collections import defaultdict
import shap
import torch
import dill
import os
import pandas as pd
import matplotlib.pyplot as plt
from .binn import BINN  
from .explainer import BINNExplainer  
from torch import nn
import pickle
import numpy as np  
from datetime import datetime


class SHAPExplainer:
    def __init__(self, input_data, device,design_matrix, model: BINN):
        with open('/data/Gene_and_network.pkl', 'rb') as file:
            data = pickle.load(file)
        connectivity_matrices_list = data
        self.B_cell_connectivity_matrices = connectivity_matrices_list['B_cell_connectivity_matrices']
        self.CD4_cell_Tfh_connectivity_matrices = connectivity_matrices_list['CD4_cell_Tfh_connectivity_matrices']
        self.CD4_cell_Th1_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th1_connectivity_matrices']
        self.CD4_cell_Th17_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th17_connectivity_matrices']
        self.CD4_cell_Th2_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th2_connectivity_matrices']
        self.CD4_cell_Treg_connectivity_matrices = connectivity_matrices_list['CD4_cell_Treg_connectivity_matrices']
        self.CD8_cell_Tem_connectivity_matrices = connectivity_matrices_list['CD8_cell_Tem_connectivity_matrices']
        self.CD8_cell_Tcm_connectivity_matrices = connectivity_matrices_list['CD8_cell_Tcm_connectivity_matrices']
        
        intersection = data["gene_list"]
        self.model = model
        self.device = device
        design_matrix =design_matrix
  
        self.fitted_input_data = self._fit_data_matrix_to_network_input(
                input_data.reset_index(),
                features=intersection,
                feature_column="Gene"
            )
        X, y, test_data, y_test= self._generate_k_folds(
                self.fitted_input_data, design_matrix=design_matrix,n_folds=3
            )
        
        self.test_data = torch.Tensor(test_data).to(self.device)
        self.background_data = torch.Tensor(X).to(self.device)



        self.y = y
        self.y_test = y_test
        self.explainer = BINNExplainer(model)
        print("SHAPExplainer initialized.")
    def explain(
        self,
        output_dir: str,
        iteration: int
     
    ):
        def get_connectivity_matrices_list():
            connectivity_matrices_list = self.model.B_cell_connectivity_matrices[:4]+ self.model.CD8_cell_Tcm_connectivity_matrices[:4]+ self.model.CD8_cell_Tem_connectivity_matrices[:4]+ self.model.CD4_cell_Th1_connectivity_matrices[:4]+self.model.CD4_cell_Th2_connectivity_matrices[:4]+ self.model.CD4_cell_Th17_connectivity_matrices[:4]+self.model.CD4_cell_Tfh_connectivity_matrices[:4]+ self.model.CD4_cell_Treg_connectivity_matrices[:4]+ self.model.B_cell_connectivity_matrices[6:10]+self.model.B_cell_connectivity_matrices[12:16]+ self.model.B_cell_connectivity_matrices[18:22]+self.model.B_cell_connectivity_matrices[24:28]+self.model.B_cell_connectivity_matrices[30:34]+self.model.CD8_cell_Tcm_connectivity_matrices[6:10]+self.model.CD8_cell_Tcm_connectivity_matrices[12:16]+  self.model.CD8_cell_Tcm_connectivity_matrices[18:22]+ self.model.CD8_cell_Tcm_connectivity_matrices[24:28]+ self.model.CD8_cell_Tcm_connectivity_matrices[30:34]+self.model.CD8_cell_Tem_connectivity_matrices[6:10]+  self.model.CD8_cell_Tem_connectivity_matrices[12:16]+  self.model.CD8_cell_Tem_connectivity_matrices[18:22]+ self.model.CD8_cell_Tem_connectivity_matrices[24:28]+ self.model.CD8_cell_Tem_connectivity_matrices[30:34]+self.model.CD4_cell_Th1_connectivity_matrices[6:10]+ self.model.CD4_cell_Th1_connectivity_matrices[12:16]+ self.model.CD4_cell_Th1_connectivity_matrices[18:22]+ self.model.CD4_cell_Th1_connectivity_matrices[24:28]+ self.model.CD4_cell_Th2_connectivity_matrices[6:10]+ self.model.CD4_cell_Th2_connectivity_matrices[12:16]+ self.model.CD4_cell_Th2_connectivity_matrices[18:22]+ self.model.CD4_cell_Th2_connectivity_matrices[24:28]+self.model.CD4_cell_Th17_connectivity_matrices[6:10]+ self.model.CD4_cell_Th17_connectivity_matrices[12:16]+ self.model.CD4_cell_Th17_connectivity_matrices[18:22]+ self.model.CD4_cell_Th17_connectivity_matrices[24:28]+self.model.CD4_cell_Tfh_connectivity_matrices[6:10]+ self.model.CD4_cell_Tfh_connectivity_matrices[12:16]+ self.model.CD4_cell_Tfh_connectivity_matrices[18:22]+ self.model.CD4_cell_Tfh_connectivity_matrices[24:28]+self.model.CD4_cell_Treg_connectivity_matrices[6:10]+ self.model.CD4_cell_Treg_connectivity_matrices[12:16]+ self.model.CD4_cell_Treg_connectivity_matrices[18:22]+ self.model.CD4_cell_Treg_connectivity_matrices[24:28]
           

            return connectivity_matrices_list

             

      
        
        
        test_data_subset = self.test_data
        background_data_subset = self.background_data
        print(test_data_subset.shape)
        print(background_data_subset.shape)
 
        y_z=np.where(self.y == 1)[0]
        print(y_z)
        current_time = datetime.now()
        print("time:", current_time)
        shap_dict, shap_dict_cell = self.explainer._explain_layer1(test_data_subset, background_data_subset,y_z,self.device)
        current_time = datetime.now()
        print("time:", current_time)

          
        feature_dict = {
                    "name": [],
                    "source name": [],
                    "target name": [],
                    "value": [],
                    "type": [],
                    "source layer":[],
                    "target layer": [],                  
            }
        feature_dict_cell = {
                    "name": [],
                    "value": [],
                    "type": [],
             
            }
        


                                
         

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
        values_cell_mean = np.mean(values_cell, axis=0)#
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
                
        for name, (first, second) in merged_dict.items():  
            shap_dict_items = {"features":[], "shap_values":[]}
            connectivity_matrices_list_items = []
            for idx in first:
                shap_dict_items["shap_values"].append(shap_dict["shap_values"][idx])
                shap_dict_items["features"].append(shap_dict["features"][idx])
                connectivity_matrices_list_items.append(connectivity_matrices_list[idx])
            result_second = values_cell_mean[second]  
            curr_layer = 0
            for sv, features, cm in zip(
                        shap_dict_items["shap_values"], shap_dict_items["features"], connectivity_matrices_list_items
                ):
     
                    sv = np.asarray(sv)   
                    sv = abs(sv)                                     
                    sv_mean = np.mean(sv, axis=0)                   
                    sv_mean_final =  sv_mean @ result_second   
            
                    for feature in range(sv_mean.shape[0]):

                        n_classes = sv_mean_final.shape[1]
                        modified_string = features[feature].replace(name+'_', '')
                        connections = cm[cm.index == modified_string]
                        connections = connections.loc[
                            :, (connections != 0).any(axis=0)
                        ]  
                        for target in connections:
                            for curr_class in range(n_classes):
                                feature_dict["name"].append(name)
                                feature_dict["source name"].append(features[feature])
                                feature_dict["target name"].append(name + "_" + target)
                                feature_dict["value"].append(sv_mean_final[feature][curr_class])
                                feature_dict["type"].append(curr_class)
                                feature_dict["source layer"].append(curr_layer)
                                feature_dict["target layer"].append(curr_layer + 1)
                                
                    curr_layer += 1
        


        
        for sv, features in zip(
                        shap_dict_cell["shap_values"], shap_dict_cell["features"]
                ):
            sv = np.asarray(sv)   
            sv = abs(sv)                                     
            sv_mean1 = np.mean(sv, axis=0)                   
            sv_mean_final1 = sv_mean1
            
            for feature in range(sv_mean1.shape[0]):

                n_classes = sv_mean_final.shape[1]
    
                for curr_class in range(n_classes):
                    feature_dict_cell["name"].append(features[feature])
                    feature_dict_cell["value"].append(sv_mean_final1[feature][curr_class])
                    feature_dict_cell["type"].append(curr_class)

        shap_df = pd.DataFrame(data=feature_dict)
        shap_csv_path = os.path.join(output_dir, f"shap_explain_iter{iteration}.csv")
        shap_df.to_csv(shap_csv_path, index=False)

        shap_df = pd.DataFrame(data=feature_dict_cell)
        shap_csv_path = os.path.join(output_dir, f"shap_explain_cell_iter{iteration}.csv")
        shap_df.to_csv(shap_csv_path, index=False)
        print(f"SHAP dictionary saved to {shap_csv_path}")

        return shap_dict
    def explain_cell(
        self,
        output_dir: str,
        iteration: int,
 
    ):
        def get_connectivity_matrices_list():
            connectivity_matrices_list = self.model.B_cell_connectivity_matrices[:4]+ self.model.CD8_cell_Tcm_connectivity_matrices[:4]+ self.model.CD8_cell_Tem_connectivity_matrices[:4]+ self.model.CD4_cell_Th1_connectivity_matrices[:4]+ self.model.CD4_cell_Th2_connectivity_matrices[:4]+ self.model.CD4_cell_Th17_connectivity_matrices[:4]+ self.model.CD4_cell_Tfh_connectivity_matrices[:4]+ self.model.CD4_cell_Treg_connectivity_matrices[:4]+ self.model.B_cell_connectivity_matrices[6:10]+ self.model.CD8_cell_Tcm_connectivity_matrices[6:10]+ self.model.CD8_cell_Tem_connectivity_matrices[6:10]+ self.model.CD4_cell_Th1_connectivity_matrices[6:10]+ self.model.CD4_cell_Th2_connectivity_matrices[6:10]+ self.model.CD4_cell_Th17_connectivity_matrices[6:10]+ self.model.CD4_cell_Tfh_connectivity_matrices[6:10]+ self.model.CD4_cell_Treg_connectivity_matrices[6:10]+ self.model.B_cell_connectivity_matrices[12:16]+ self.model.CD8_cell_Tcm_connectivity_matrices[12:16]+ self.model.CD8_cell_Tem_connectivity_matrices[12:16]+ self.model.CD4_cell_Th1_connectivity_matrices[12:16]+ self.model.CD4_cell_Th2_connectivity_matrices[12:16]+ self.model.CD4_cell_Th17_connectivity_matrices[12:16]+ self.model.CD4_cell_Tfh_connectivity_matrices[12:16]+ self.model.CD4_cell_Treg_connectivity_matrices[12:16]+ self.model.B_cell_connectivity_matrices[18:22]+ self.model.CD8_cell_Tcm_connectivity_matrices[18:22]+ self.model.CD8_cell_Tem_connectivity_matrices[18:22]+ self.model.CD4_cell_Th1_connectivity_matrices[18:22]+ self.model.CD4_cell_Th2_connectivity_matrices[18:22]+ self.model.CD4_cell_Th17_connectivity_matrices[18:22]+ self.model.CD4_cell_Tfh_connectivity_matrices[18:22]+ self.model.CD4_cell_Treg_connectivity_matrices[18:22]+ self.model.B_cell_connectivity_matrices[24:28]+ self.model.CD8_cell_Tcm_connectivity_matrices[24:28]+ self.model.CD8_cell_Tem_connectivity_matrices[24:28]+ self.model.CD4_cell_Th1_connectivity_matrices[24:28]+ self.model.CD4_cell_Th2_connectivity_matrices[24:28]+ self.model.CD4_cell_Th17_connectivity_matrices[24:28]+ self.model.CD4_cell_Tfh_connectivity_matrices[24:28]+ self.model.CD4_cell_Treg_connectivity_matrices[24:28]+ self.model.B_cell_connectivity_matrices[30:34]+ self.model.CD8_cell_Tcm_connectivity_matrices[30:34]+ self.model.CD8_cell_Tem_connectivity_matrices[30:34]

            return connectivity_matrices_list

             



        test_data_subset = self.test_data
        background_data_subset = self.background_data

     
        current_time = datetime.now()
        print("time:", current_time)
        y = self.y
        shap_dict_cell = self.explainer._explain_cell_layer(test_data_subset,y, background_data_subset)
        current_time = datetime.now()
        print("time:", current_time)


        feature_dict_cell = {
                    "name": [],
                    "value": [],
                    "type": [],
             
            }
                
        
        for sv, features in zip(
                        shap_dict_cell["shap_values"], shap_dict_cell["features"]
                ):
            sv = np.asarray(sv)   
            sv = abs(sv)                                     
            sv_mean1 = np.mean(sv, axis=0)                   
            sv_mean_final1 = sv_mean1
            
            for feature in range(sv_mean1.shape[0]):

                n_classes = sv_mean_final1.shape[1]
    
                for curr_class in range(n_classes):
                    feature_dict_cell["name"].append(features[feature])
                    feature_dict_cell["value"].append(sv_mean_final1[feature][curr_class])
                    feature_dict_cell["type"].append(curr_class)

        shap_df = pd.DataFrame(data=feature_dict_cell)
        shap_csv_path = os.path.join(output_dir, f"shap_cell_iter{iteration}.csv")
        shap_df.to_csv(shap_csv_path, index=False)
        print(f"SHAP dictionary saved to {shap_csv_path}")
        #torch.cuda.empty_cache()
        return shap_dict_cell
    def _fit_data_matrix_to_network_input(
    self, data_matrix: pd.DataFrame, features, feature_column="Gene"
        ) -> pd.DataFrame:
        nr_features_in_matrix = len(data_matrix.index)
        if len(features) > nr_features_in_matrix:
            features_df = pd.DataFrame(features, columns=[feature_column])
            data_matrix = data_matrix.merge(features_df, how="right", on=feature_column)
        if len(features) > 0:
            data_matrix.set_index(feature_column, inplace=True)
            features = list(features)
            data_matrix = data_matrix.loc[features]
        return data_matrix
    def _generate_k_folds(
        self,
        data_matrix: pd.DataFrame,
        design_matrix: pd.DataFrame,
        groups=list(range(1, 25)), 
        n_folds=3,
        test_size=0.15, 
        val_size=0.15,  
        random_state=42
    ):
        y = []
        dfs = []
        for i, group in enumerate(groups):
            group_samples = design_matrix[design_matrix["group"] == group]["sample"].values
            df_group = data_matrix[group_samples].T 

            dfs.append(df_group)
            y += [group-1 for _ in group_samples]   
        group_samples = design_matrix[design_matrix["group"] == 2]["sample"].values
        test_data = data_matrix[group_samples].T.fillna(0).to_numpy()
        y_test = group_samples
        y = np.array(y)
        X = pd.concat(dfs).fillna(0).to_numpy()
     


        
        return X, y,test_data,y_test




