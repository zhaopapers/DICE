import shap
import numpy as np
import torch
from .binn import BINN
import pandas as pd
import lightning.pytorch as pl
from .feature_selection import RecursivePathwayElimination
from torch import nn
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import gc
import os
import dill
class BINNExplainer:
    """
    A class for explaining the predictions of a BINN model using SHAP values.

    Args:
        model (BINN): A trained BINN model.
    """
    
    def __init__(self, model: BINN):
        self.model = model
        self.model.eval()
        self.device = model.device
        self.batchnorm_np = nn.BatchNorm1d(7057, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(self.device)
        if multiprocessing.get_start_method(allow_none=True) is None:
            multiprocessing.set_start_method("spawn")

    def _forward_residual_subnet(self, subnet, x: torch.Tensor) -> torch.Tensor:
        
        x =x.to("cuda:0")
        for name, layer in subnet.named_children():
                x = layer(x)          
        return x
    def _forward_residual(self, x: torch.Tensor):
        x_final = torch.tensor([], device=self.device)
        x = x.to(self.device)
        Gene_input = self.batchnorm_np(x)
        gene_in_x_list = []
        for name, layer in self.cell_layers.named_children():
            if name.startswith("differentiation"):
                x = layer(x)
                x_final = torch.cat((x_final, x), dim=1)
            elif name.startswith("Gene_in"):
                x = layer(x)
                x = self.batchnorm_np(x)
                x = (Gene_input + x) / 2
                y = self.cell_layers[2](x)
                y = self.cell_layers[3](y)
                gene_output = y
                gene_in_x_list.append(gene_output.detach().cpu().numpy())
            else:
                x = layer(x)
        return x_final, gene_in_x_list
    def _forward_long_subnet(self,subset, x: torch.Tensor,batchnorm_np,device) -> torch.Tensor:
        x_final = torch.tensor([], device=device)
        x = x.to(device)
        Gene_input = x.to(device)
        gene_output = None
        gene_outputs_list = []
        for name, layer in subset.named_children():
            if name.startswith("differentiation"):
                x = layer(x)
                x_final = torch.cat((x_final, x), dim=1)
            elif name.startswith("Gene_in"):
                x = layer(x)  
                test_data = batchnorm_np(Gene_input)
                x = (test_data + x) / 2
                y = subset[index+1](x)  
                y = subset[index+2](y)        
                gene_output = y
                gene_outputs_list.append(gene_output.detach().cpu().numpy())
            else:
                x = layer(x)
        return gene_outputs_list,x_final       

                      
    def _process_subnet(subnet_name, subnet,device,background_data, test_data, layer_names,layer_cell_types,x_final_list_back,x_final_list_test,x_gene_list_back,x_gene_list_test):

        try:
            batchnorm_np = nn.BatchNorm1d(7057, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)
            subnet = subnet.to(device)
            layer_name = layer_names[subnet_name]
            feature_index = 0
            background_data = background_data.to(device)
            test_data = test_data.to(device)
            partial_shap_dict = {"features": [], "shap_values": []}
            intermediate_data = test_data.to(device)
            if subnet_name in ["HSC_B_cell_layers", "HSC_CD8_cell_Tcm_layers", "HSC_CD8_cell_Tem_layers", "HSC_CD4_cell_Th1_layers", "HSC_CD4_cell_Th2_layers", "HSC_CD4_cell_Th17_layers", "HSC_CD4_cell_Tfh_layers", "HSC_CD4_cell_Treg_layers"]:
                x_final=None
                #print(layer_name)       
                for name, layer in subnet.named_children():
                    if isinstance(layer, nn.Linear):      
                        explainer = shap.DeepExplainer(
                                        (subnet, layer), 
                                        background_data 
                                    )
                        shap_values = explainer.shap_values(test_data, check_additivity=False) 
                        partial_shap_dict["features"].append([f"{subnet_name}_{item}" for item in layer_name[feature_index]])
                        partial_shap_dict["shap_values"].append(shap_values)
                        feature_index += 1
                        intermediate_data = layer(intermediate_data)
            
            elif subnet_name in ["B_cell_layers", "CD8_cell_Tcm_layers", "CD8_cell_Tem_layers", "CD4_cell_Th1_layers", "CD4_cell_Th2_layers", "CD4_cell_Th17_layers", "CD4_cell_Tfh_layers", "CD4_cell_Treg_layers"]:
                subnet_name_ = layer_cell_types[subnet_name]
                Gene_list_back =x_gene_list_back[subnet_name]
                Gene_list_test =x_gene_list_test[subnet_name]
                for i in range(len(Gene_list_back)):
                    subnet_name_subset = subnet_name_[i]
                    Gene_back = Gene_list_back[i]
                    Gene_test = Gene_list_test[i]
                    feature_index = 0
                    intermediate_data_back = Gene_back
                    intermediate_data_test = Gene_test                  
                    model_list = nn.ModuleList(list(subnet.children()))   
                    subnet_su = nn.Sequential(*model_list[23*(i+1):(23*(i+1)+16)])
                    layer_name = layer_names[subnet_name]                       
                    layer_name = layer_name[6*(i+1):(6*(i+1)+4)]
                    #print(layer_name)
                    back_data_tensor = torch.from_numpy(intermediate_data_back).float()                  
                    test_data_tensor = torch.from_numpy(intermediate_data_test).float()  

# If using GPU, move tensor to the same device as the model
                    back_data_tensor = back_data_tensor.to(device)                    
                    test_data_tensor = test_data_tensor.to(device)
                    subnet_su = subnet_su.to(device)  
                    intermediate_data_back = back_data_tensor        
                    intermediate_data_test = test_data_tensor
                    for name, layer in subnet_su.named_children():
                        if isinstance(layer, nn.Linear):    
                            explainer = shap.DeepExplainer(
                                                (subnet_su, layer), 
                                                back_data_tensor  
                                            )
                            shap_values = explainer.shap_values(test_data_tensor, check_additivity=False) 
                            partial_shap_dict["features"].append([f"{subnet_name_subset}_{item}" for item in layer_name[feature_index]])
                            partial_shap_dict["shap_values"].append(shap_values)
                            feature_index += 1
                            intermediate_data_back = layer(intermediate_data_back)
                            intermediate_data_test = layer(intermediate_data_test)
            return partial_shap_dict
                    
        finally:
            gc.collect()
            torch.cuda.empty_cache()
    from collections import defaultdict
    def shap_single_GO(shap_dict,samples,connectivity_matrices_list,target_key):
    
        feature_dict = {
                "name": [],
                "source name": [],
                "target name": [], 
                "type": [],
                "source layer":[],
                "target layer": [],                  
            }
        for sample in samples:
            feature_dict[sample] = []  
        feature_id_mapping = {}
        feature_id = 0
        feature_id_mapping["root"] = feature_id
        for layer_features in shap_dict["features"]:
            for feature in layer_features:
                feature_id += 1
                feature_id_mapping[feature] = feature_id
    
        feature_id_mapping = {}
        feature_id = 0
        feature_id_mapping["root"] = feature_id
        for layer_features in shap_dict["features"]:
            for feature in layer_features:
                feature_id += 1
                feature_id_mapping[feature] = feature_id
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
        output_file = '/model/shap_GO_teff_Tcm_0.csv'
        df.to_csv(output_file, index=False)
        return df

    def shap_single_cell(shap_dict,samples):
        feature_dict = {
                        "name": [],
                        "type": [],  
                }
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
            output_file = '{}shap_cell.csv'.format(dir)
            df.to_csv(output_file, index=False)
    def _explain_layer(
            self, background_data: torch.Tensor, test_data: torch.Tensor,device
        ) -> dict:
            self.model = self.model.to(device)
            test_data = test_data.to(device)
            background_data = background_data.to(device)
            print(test_data.shape)
            print(background_data.shape)
            def deep_model_list(deep_cell_name,num):
                model_list = nn.ModuleList(list(deep_cell_name.children()))   
                layers = nn.Sequential(*model_list[:num])
                return layers
            subnets = {
                'HSC_B_cell_layers': deep_model_list(self.model.B_cell_layers,16),
                'HSC_CD8_cell_Tcm_layers': deep_model_list(self.model.CD8_cell_Tcm_layers,16),
                'HSC_CD8_cell_Tem_layers': deep_model_list(self.model.CD8_cell_Tem_layers,16),
                'HSC_CD4_cell_Th1_layers': deep_model_list(self.model.CD4_cell_Th1_layers,16),
                'HSC_CD4_cell_Th2_layers': deep_model_list(self.model.CD4_cell_Th2_layers,16),
                'HSC_CD4_cell_Th17_layers': deep_model_list(self.model.CD4_cell_Th17_layers,16),
                'HSC_CD4_cell_Tfh_layers': deep_model_list(self.model.CD4_cell_Tfh_layers,16),
                'HSC_CD4_cell_Treg_layers': deep_model_list(self.model.CD4_cell_Treg_layers,16),
                'B_cell_layers': self.model.B_cell_layers,
                'CD8_cell_Tcm_layers': self.model.CD8_cell_Tcm_layers,
                'CD8_cell_Tem_layers': self.model.CD8_cell_Tem_layers,
                'CD4_cell_Th1_layers': self.model.CD4_cell_Th1_layers,
                'CD4_cell_Th2_layers': self.model.CD4_cell_Th2_layers,
                'CD4_cell_Th17_layers': self.model.CD4_cell_Th17_layers,
                'CD4_cell_Tfh_layers': self.model.CD4_cell_Tfh_layers,
                'CD4_cell_Treg_layers': self.model.CD4_cell_Treg_layers
            }

            layer_names = {
                'HSC_B_cell_layers': self.model.B_cell_layer_names[:4],
                'HSC_CD8_cell_Tcm_layers': self.model.CD8_cell_Tcm_layer_names[:4],
                'HSC_CD8_cell_Tem_layers': self.model.CD8_cell_Tem_layer_names[:4],
                'HSC_CD4_cell_Th1_layers': self.model.CD4_cell_Th1_layer_names[:4],
                'HSC_CD4_cell_Th2_layers': self.model.CD4_cell_Th2_layer_names[:4],
                'HSC_CD4_cell_Th17_layers': self.model.CD4_cell_Th17_layer_names[:4],
                'HSC_CD4_cell_Tfh_layers': self.model.CD4_cell_Tfh_layer_names[:4],
                'HSC_CD4_cell_Treg_layers': self.model.CD4_cell_Treg_layer_names[:4],
                'B_cell_layers': self.model.B_cell_layer_names,
                'CD8_cell_Tcm_layers': self.model.CD8_cell_Tcm_layer_names,
                'CD8_cell_Tem_layers': self.model.CD8_cell_Tem_layer_names,
                'CD4_cell_Th1_layers': self.model.CD4_cell_Th1_layer_names,
                'CD4_cell_Th2_layers': self.model.CD4_cell_Th2_layer_names,
                'CD4_cell_Th17_layers': self.model.CD4_cell_Th17_layer_names,
                'CD4_cell_Tfh_layers': self.model.CD4_cell_Tfh_layer_names,
                'CD4_cell_Treg_layers': self.model.CD4_cell_Treg_layer_names
            }
            layer_cell_types = {
                'B_cell_layers': ["CLP_B_cell_layers","Pro_B_cell_layers","Pre_B_cell_layers","immature_B_cell_layers","mature_B_cell_layers"],
                'CD8_cell_Tcm_layers': ["CLP_CD8_cell_Tcm_layers","Pro_CD8_cell_Tcm_layers","Pre_CD8_cell_Tcm_layers","naive_CD8_cell_Tcm_layers","Teff_CD8_cell_Tcm_layers"],
                'CD8_cell_Tem_layers': ["CLP_CD8_cell_Tem_layers","Pro_CD8_cell_Tem_layers","Pre_CD8_cell_Tem_layers","naive_CD8_cell_Tem_layers","Teff_CD8_cell_Tem_layers"],
                'CD4_cell_Th1_layers': ["CLP_CD4_cell_Th1_layers","Pro_CD4_cell_Th1_layers","Pre_CD4_cell_Th1_layers","naive_CD4_cell_Th1_layers"],
                'CD4_cell_Th2_layers': ["CLP_CD4_cell_Th2_layers","Pro_CD4_cell_Th2_layers","Pre_CD4_cell_Th2_layers","naive_CD4_cell_Th2_layers"],
                'CD4_cell_Th17_layers': ["CLP_CD4_cell_Th17_layers","Pro_CD4_cell_Th17_layers","Pre_CD4_cell_Th17_layers","naive_CD4_cell_Th17_layers"],
                'CD4_cell_Tfh_layers': ["CLP_CD4_cell_Tfh_layers","Pro_CD4_cell_Tfh_layers","Pre_CD4_cell_Tfh_layers","naive_CD4_cell_Tfh_layers"],
                'CD4_cell_Treg_layers': ["CLP_CD4_cell_Treg_layers","Pro_CD4_cell_Treg_layers","Pre_CD4_cell_Treg_layers","naive_CD4_cell_Treg_layers"]
            }
            shap_dict = {"features": [], "shap_values": []}
            shap_dict_cell = {"features": [], "shap_values": []}
            shap_dict = {"features": [], "shap_values": []}
            x_final_list_back = {
            
            "B_cell_layers":[],    

            "CD8_cell_Tcm_layers":[],
          
            "CD8_cell_Tem_layers":[],
          
            "CD4_cell_Th1_layers":[],
        
            "CD4_cell_Th2_layers":[],
          
            "CD4_cell_Th17_layers":[],
        
            "CD4_cell_Tfh_layers":[],
         
            "CD4_cell_Treg_layers":[]
            }
            x_gene_list_back = {
            
            "B_cell_layers":[],    
            
            "CD8_cell_Tcm_layers":[],
            
            "CD8_cell_Tem_layers":[],
       
            "CD4_cell_Th1_layers":[],
          
            "CD4_cell_Th2_layers":[],
          
            "CD4_cell_Th17_layers":[],
          
            "CD4_cell_Tfh_layers":[],
            
            "CD4_cell_Treg_layers":[]
            }     
            x_final_list_test = {
            
            "B_cell_layers":[],    
            
            "CD8_cell_Tcm_layers":[],
          
            "CD8_cell_Tem_layers":[],
          
            "CD4_cell_Th1_layers":[],
        
            "CD4_cell_Th2_layers":[],
          
            "CD4_cell_Th17_layers":[],
        
            "CD4_cell_Tfh_layers":[],
         
            "CD4_cell_Treg_layers":[]
            }
            x_gene_list_test = {
            
            "B_cell_layers":[],    
            
            "CD8_cell_Tcm_layers":[],
            
            "CD8_cell_Tem_layers":[],
       
            "CD4_cell_Th1_layers":[],
          
            "CD4_cell_Th2_layers":[],
          
            "CD4_cell_Th17_layers":[],
          
            "CD4_cell_Tfh_layers":[],
            
            "CD4_cell_Treg_layers":[]
            }          
            shap_dict = {"features": [], "shap_values": []}
            #combined_output = torch.cat([self._forward_residual_subnet(subnet, test_data) for subnet in subnets.values()], dim=1)
            combined_output_test = x_final_list_test 
            combined_output_back = x_final_list_back 
            self.model._forward_long_subnet = self._forward_long_subnet
            feature_info = []
            batchnorm_np = nn.BatchNorm1d(7057, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(device)

            cell_final_layer_name = []
      
            for subnet_name, subnet in subnets.items():
             
                subnet = subnets[subnet_name.replace("HSC_", "")] 
                if subnet_name =="HSC_B_cell_layers": 
                    
                    with torch.no_grad():
                        Gene_list_back, x_final_back= self.model._forward_long_subnet(subnet,background_data.detach(),batchnorm_np,device)
                        Gene_list_test, x_final_test= self.model._forward_long_subnet(subnet,test_data.detach(),batchnorm_np,device)                        
                    key = subnet_name.replace("HSC_", "")
                    x_final_list_back[key] = x_final_back
                    x_gene_list_back[key]=Gene_list_back                    
                    x_final_list_test[key] = x_final_test
                    x_gene_list_test[key]=Gene_list_test
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                        
                elif subnet_name == "HSC_CD8_cell_Tcm_layers":
                    with torch.no_grad():
                        self.model.cell_layers = self.model.CD8_cell_Tcm_layers
                        Gene_list_back, x_final_back= self.model._forward_long_subnet(subnet,background_data.detach(),batchnorm_np,device)
                        Gene_list_test, x_final_test= self.model._forward_long_subnet(subnet,test_data.detach(),batchnorm_np,device)   
                    key = subnet_name.replace("HSC_", "")
                    x_final_list_back[key] = x_final_back
                    x_gene_list_back[key]=Gene_list_back                    
                    x_final_list_test[key] = x_final_test
                    x_gene_list_test[key]=Gene_list_test
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                        
                elif subnet_name == "HSC_CD8_cell_Tem_layers":
                    with torch.no_grad():
                        self.model.cell_layers = self.model.CD8_cell_Tem_layers
                        Gene_list_back, x_final_back= self.model._forward_long_subnet(subnet,background_data.detach(),batchnorm_np,device)
                        Gene_list_test, x_final_test= self.model._forward_long_subnet(subnet,test_data.detach(),batchnorm_np,device)   
                    key = subnet_name.replace("HSC_", "")                    
                    x_final_list_back[key] = x_final_back
                    x_gene_list_back[key]=Gene_list_back                    
                    x_final_list_test[key] = x_final_test
                    x_gene_list_test[key]=Gene_list_test
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                                                                    
                elif subnet_name == "HSC_CD4_cell_Th1_layers":
                    with torch.no_grad():
                        self.model.cell_layers = self.model.CD4_cell_Th1_layers
                        Gene_list_back, x_final_back= self.model._forward_long_subnet(subnet,background_data.detach(),batchnorm_np,device)
                        Gene_list_test, x_final_test= self.model._forward_long_subnet(subnet,test_data.detach(),batchnorm_np,device)   
                    key = subnet_name.replace("HSC_", "")                       
                    x_final_list_back[key] = x_final_back
                    x_gene_list_back[key]=Gene_list_back                    
                    x_final_list_test[key] = x_final_test
                    x_gene_list_test[key]=Gene_list_test
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                                           
                elif subnet_name == "HSC_CD4_cell_Th2_layers":
                    with torch.no_grad():
                        self.model.cell_layers = self.model.CD4_cell_Th2_layers
                        Gene_list_back, x_final_back= self.model._forward_long_subnet(subnet,background_data.detach(),batchnorm_np,device)
                        Gene_list_test, x_final_test= self.model._forward_long_subnet(subnet,test_data.detach(),batchnorm_np,device)   
                    key = subnet_name.replace("HSC_", "") 
                    x_final_list_back[key] = x_final_back
                    x_gene_list_back[key]=Gene_list_back                    
                    x_final_list_test[key] = x_final_test
                    x_gene_list_test[key]=Gene_list_test
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                                              
                elif subnet_name ==  "HSC_CD4_cell_Th17_layers":
                    with torch.no_grad():
                        self.model.cell_layers = self.model.CD4_cell_Th17_layers
                        Gene_list_back, x_final_back= self.model._forward_long_subnet(subnet,background_data.detach(),batchnorm_np,device)
                        Gene_list_test, x_final_test= self.model._forward_long_subnet(subnet,test_data.detach(),batchnorm_np,device)   
                    key = subnet_name.replace("HSC_", "") 
                    x_final_list_back[key] = x_final_back
                    x_gene_list_back[key]=Gene_list_back                    
                    x_final_list_test[key] = x_final_test
                    x_gene_list_test[key]=Gene_list_test
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                        
                elif subnet_name ==  "HSC_CD4_cell_Tfh_layers":
                    with torch.no_grad():
                        self.model.cell_layers = self.model.CD4_cell_Tfh_layers
                        Gene_list_back, x_final_back= self.model._forward_long_subnet(subnet,background_data.detach(),batchnorm_np,device)
                        Gene_list_test, x_final_test= self.model._forward_long_subnet(subnet,test_data.detach(),batchnorm_np,device)   
                    key = subnet_name.replace("HSC_", "") 
                    x_final_list_back[key] = x_final_back
                    x_gene_list_back[key]=Gene_list_back                    
                    x_final_list_test[key] = x_final_test
                    x_gene_list_test[key]=Gene_list_test
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])
                elif subnet_name ==   "HSC_CD4_cell_Treg_layers":
                    with torch.no_grad():
                        self.model.cell_layers = self.model.CD4_cell_Treg_layers
                        Gene_list_back, x_final_back= self.model._forward_long_subnet(subnet,background_data.detach(),batchnorm_np,device)
                        Gene_list_test, x_final_test= self.model._forward_long_subnet(subnet,test_data.detach(),batchnorm_np,device)   
                    key = subnet_name.replace("HSC_", "") 
                    x_final_list_back[key] = x_final_back
                    x_gene_list_back[key]=Gene_list_back                    
                    x_final_list_test[key] = x_final_test
                    x_gene_list_test[key]=Gene_list_test
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])
                else:
                    continue
            with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn"), max_workers=5) as executor:
                futures = []
                for subnet_name, subnet in subnets.items():
                    futures.append(executor.submit(
                        BINNExplainer._process_subnet, subnet_name, subnet,device,background_data, test_data, layer_names,layer_cell_types, x_final_list_back,x_final_list_test,x_gene_list_back,x_gene_list_test
                    ))
                for future in futures:
                    partial_shap_dict= future.result()
                    
                    shap_dict["features"].extend(partial_shap_dict["features"])
                    shap_dict["shap_values"].extend(partial_shap_dict["shap_values"])
            keys_order = [
                          "B_cell_layers",
                        
                        "CD8_cell_Tcm_layers",
                     
                        "CD8_cell_Tem_layers",
                     
                        "CD4_cell_Th1_layers",
                      
                        "CD4_cell_Th2_layers",
                  
                        "CD4_cell_Th17_layers",
                     
                        "CD4_cell_Tfh_layers",
                    
                        "CD4_cell_Treg_layers"
                        ]  
            outputs = []
            outputs = [x_final_list_test[key] for key in keys_order]    
            for i, output in enumerate(outputs):
                if isinstance(output, list):
                    outputs[i] = torch.tensor(output).to(device)
            combined_output = torch.cat(outputs, dim=1)
            outputs_back = []
            outputs_back = [x_final_list_back[key] for key in keys_order]    
            print(outputs_back)
            for i, output in enumerate(outputs_back):
                if isinstance(output, list):
                    outputs_back[i] = torch.tensor(output).to(device)
            combined_output_back = torch.cat(outputs_back, dim=1)
            print(combined_output_back.shape) 
            outputs_test = []
            outputs_test = [x_final_list_test[key] for key in keys_order]    
            for i, output in enumerate(outputs_test):

                if isinstance(output, list):
                    outputs_test[i] = torch.tensor(output).to(device)
            combined_output_test = torch.cat(outputs_test, dim=1)
            
            for name, layer in self.model.final_layers.named_children():
                if isinstance(layer, nn.Linear):
                    explainer = shap.DeepExplainer((self.model.final_layers, layer), combined_output_back)
                    shap_values = explainer.shap_values(combined_output_test, check_additivity=False)
                    
                    shap_dict_cell["features"].append(cell_final_layer_name)
                    shap_dict_cell["shap_values"].append(shap_values)
                    shap_dict["features"].append(cell_final_layer_name)
                    shap_dict["shap_values"].append(shap_values)
                 
                   
                    combined_output = layer(combined_output)

                elif isinstance(layer, (nn.Tanh, nn.ReLU, nn.LeakyReLU)):
                    
                    combined_output = layer(combined_output)


           
            
            return shap_dict, shap_dict_cell
    def _explain_cell_layer(
            self, background_data: torch.Tensor,y, test_data: torch.Tensor
        ) -> dict:

            
            self.model = self.model.to("cuda:1")
                    
            test_data = background_data.to("cuda:1")
            background_data = background_data.to("cuda:1")
            def deep_model_list(deep_cell_name,num):
                model_list = nn.ModuleList(list(deep_cell_name.children()))   
                layers = nn.Sequential(*model_list[:num])
                return layers

            subnets = {
                
                'HSC_B_cell_layers': deep_model_list(self.model.B_cell_layers,16),
                'HSC_CD8_cell_Tcm_layers': deep_model_list(self.model.CD8_cell_Tcm_layers,16),
                'HSC_CD8_cell_Tem_layers': deep_model_list(self.model.CD8_cell_Tem_layers,16),
                'HSC_CD4_cell_Th1_layers': deep_model_list(self.model.CD4_cell_Th1_layers,16),
                'HSC_CD4_cell_Th2_layers': deep_model_list(self.model.CD4_cell_Th2_layers,16),
                'HSC_CD4_cell_Th17_layers': deep_model_list(self.model.CD4_cell_Th17_layers,16),
                'HSC_CD4_cell_Tfh_layers': deep_model_list(self.model.CD4_cell_Tfh_layers,16),
                'HSC_CD4_cell_Treg_layers': deep_model_list(self.model.CD4_cell_Treg_layers,16),
                'B_cell_layers': self.model.B_cell_layers,
                'CD8_cell_Tcm_layers': self.model.CD8_cell_Tcm_layers,
                'CD8_cell_Tem_layers': self.model.CD8_cell_Tem_layers,
                'CD4_cell_Th1_layers': self.model.CD4_cell_Th1_layers,
                'CD4_cell_Th2_layers': self.model.CD4_cell_Th2_layers,
                'CD4_cell_Th17_layers': self.model.CD4_cell_Th17_layers,
                'CD4_cell_Tfh_layers': self.model.CD4_cell_Tfh_layers,
                'CD4_cell_Treg_layers': self.model.CD4_cell_Treg_layers


            }


  

            layer_cell_types = {
                'B_cell_layers': ["CLP_B_cell_layers","Pro_B_cell_layers","Pre_B_cell_layers","immature_B_cell_layers","mature_B_cell_layers"],
                'CD8_cell_Tcm_layers': ["CLP_CD8_cell_Tcm_layers","Pro_CD8_cell_Tcm_layers","Pre_CD8_cell_Tcm_layers","naive_CD8_cell_Tcm_layers","Teff_CD8_cell_Tcm_layers"],
                'CD8_cell_Tem_layers': ["CLP_CD8_cell_Tem_layers","Pro_CD8_cell_Tem_layers","Pre_CD8_cell_Tem_layers","naive_CD8_cell_Tem_layers","Teff_CD8_cell_Tem_layers"],
                'CD4_cell_Th1_layers': ["CLP_CD4_cell_Th1_layers","Pro_CD4_cell_Th1_layers","Pre_CD4_cell_Th1_layers","naive_CD4_cell_Th1_layers"],
                'CD4_cell_Th2_layers': ["CLP_CD4_cell_Th2_layers","Pro_CD4_cell_Th2_layers","Pre_CD4_cell_Th2_layers","naive_CD4_cell_Th2_layers"],
                'CD4_cell_Th17_layers': ["CLP_CD4_cell_Th17_layers","Pro_CD4_cell_Th17_layers","Pre_CD4_cell_Th17_layers","naive_CD4_cell_Th17_layers"],
                'CD4_cell_Tfh_layers': ["CLP_CD4_cell_Tfh_layers","Pro_CD4_cell_Tfh_layers","Pre_CD4_cell_Tfh_layers","naive_CD4_cell_Tfh_layers"],
                'CD4_cell_Treg_layers': ["CLP_CD4_cell_Treg_layers","Pro_CD4_cell_Treg_layers","Pre_CD4_cell_Treg_layers","naive_CD4_cell_Treg_layers"]
            }
            
            shap_dict_cell = {"features": [], "shap_values": []}

            x_final_list = {
            
            "B_cell_layers":[],    
            
            "CD8_cell_Tcm_layers":[],
          
            "CD8_cell_Tem_layers":[],
          
            "CD4_cell_Th1_layers":[],
        
            "CD4_cell_Th2_layers":[],
          
            "CD4_cell_Th17_layers":[],
        
            "CD4_cell_Tfh_layers":[],
         
            "CD4_cell_Treg_layers":[]
            }      

            combined_output = x_final_list          
            batchnorm_np = nn.BatchNorm1d(7057, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to("cuda:1")

            cell_final_layer_name = []
      
            for subnet_name, subnet in subnets.items():
          
                subnet = subnets[subnet_name.replace("HSC_", "")] 
                if subnet_name =="HSC_B_cell_layers": 
                    self.model.cell_layers = self.model.B_cell_layers
                    with torch.no_grad():
                        x_final= self.model._forward_residual(test_data)
                    key = subnet_name.replace("HSC_", "")
                    x_final_list[key] = x_final
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])
                elif subnet_name == "HSC_CD8_cell_Tcm_layers":
                    self.model.cell_layers = self.model.CD8_cell_Tcm_layers
                    with torch.no_grad():
                        x_final= self.model._forward_residual(test_data)
                    key = subnet_name.replace("HSC_", "")
                    x_final_list[key] = x_final
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                        
                elif subnet_name == "HSC_CD8_cell_Tem_layers":
                    self.model.cell_layers = self.model.CD8_cell_Tem_layers
                    with torch.no_grad():
                        x_final= self.model._forward_residual(test_data)
                    key = subnet_name.replace("HSC_", "")                    
                    x_final_list[key] = x_final
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                                                                    
                elif subnet_name == "HSC_CD4_cell_Th1_layers":
                    self.model.cell_layers = self.model.CD4_cell_Th1_layers
                    with torch.no_grad():
                        x_final= self.model._forward_residual(test_data)
                    key = subnet_name.replace("HSC_", "")                       
                    x_final_list[key] = x_final
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                                           
                elif subnet_name == "HSC_CD4_cell_Th2_layers":
                    self.model.cell_layers = self.model.CD4_cell_Th2_layers                    
                    with torch.no_grad():
                        x_final= self.model._forward_residual(test_data)
                    key = subnet_name.replace("HSC_", "") 
                    x_final_list[key] = x_final
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                        
                                        
                elif subnet_name ==  "HSC_CD4_cell_Th17_layers":
                    self.model.cell_layers = self.model.CD4_cell_Th17_layers
                    with torch.no_grad():
                        x_final= self.model._forward_residual(test_data)
                    key = subnet_name.replace("HSC_", "") 
                    x_final_list[key] = x_final
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])                        
                elif subnet_name ==  "HSC_CD4_cell_Tfh_layers":
                    self.model.cell_layers = self.model.CD4_cell_Tfh_layers
                    with torch.no_grad(): 
                        x_final= self.model._forward_residual(test_data)
                    key = subnet_name.replace("HSC_", "") 
                    x_final_list[key] = x_final
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])
                elif subnet_name ==   "HSC_CD4_cell_Treg_layers":
                    self.model.cell_layers = self.model.CD4_cell_Treg_layers                    
                    with torch.no_grad():
                        x_final= self.model._forward_residual(test_data)
                    key = subnet_name.replace("HSC_", "") 
                    x_final_list[key] = x_final
                    current_list = layer_cell_types[key]
                    cell_final_layer_name = cell_final_layer_name + ([subnet_name] * 16)
                    cell_final_layer_name.extend([element for element in current_list for _ in range(16)])
                else:
                    continue
            keys_order = [
                          "B_cell_layers",
                        
                        "CD8_cell_Tcm_layers",
                     
                        "CD8_cell_Tem_layers",
                     
                        "CD4_cell_Th1_layers",
                      
                        "CD4_cell_Th2_layers",
                  
                        "CD4_cell_Th17_layers",
                     
                        "CD4_cell_Tfh_layers",
                    
                        "CD4_cell_Treg_layers"
                        ]  
            outputs = []

            outputs = [x_final_list[key] for key in keys_order]

            for i, output in enumerate(outputs):

                if isinstance(output, list):
                    outputs[i] = torch.tensor(output).to("cuda:1")
            combined_output = torch.cat(outputs, dim=1)
            
            for name, layer in self.model.final_layers.named_children():
                if isinstance(layer, nn.Linear):
                    explainer = shap.DeepExplainer((self.model.final_layers, layer), combined_output)
                    shap_values = explainer.shap_values(combined_output, check_additivity=False)                  
                    shap_dict_cell["features"].append(cell_final_layer_name)
                    shap_dict_cell["shap_values"].append(shap_values)
                    combined_output = layer(combined_output)

                elif isinstance(layer, (nn.Tanh, nn.ReLU, nn.LeakyReLU)):
                    combined_output = layer(combined_output)   
            return shap_dict_cell
            
