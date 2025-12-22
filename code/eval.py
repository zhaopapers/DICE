import pandas as pd
import dill
import pickle
import torch
import numpy as np
from binn import BINN
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
class evaluate:
    def __init__(self, input_data, design_matrix, feature, model: BINN):
        self.model = model
        with open('/home/zxl/hdd/cellfate/data/Gene_and_network1.pkl', 'rb') as file:
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
        

        design_matrix =design_matrix
  
        self.fitted_input_data = self._fit_data_matrix_to_network_input(
                input_data.reset_index(),
                
                feature_column="Gene",
                feature = feature
            )
        X,y = self._generate_k_folds(
                self.fitted_input_data, design_matrix=design_matrix,n_folds=3
            )

        self.test_data = torch.Tensor(X).to("cpu")
        tensor= torch.Tensor(y).to("cpu")
       
        tensor = tensor.to(torch.int) 

        y = tensor.long() 
        model.eval()
        with torch.no_grad():
            predictions = model(self.test_data)
        predicted_labels = torch.argmax(predictions, dim=1)




        print("y shape:", y)
        print("predictions shape:", predictions.shape)
        print("predicted_labels shape:", predicted_labels.shape)
        print("y:", y)
        print("predicted_labels:", predicted_labels)
        print("y.unique",y.unique())
        print("predicted_labels.unique",predicted_labels.unique())

        y_cpu = y.cpu()

        acc = self.calculate_accuracy(y, predicted_labels)
        f1 = self.calculate_F1(y, predicted_labels)
        predicted_labels_cpu = predicted_labels.cpu()


        y_np = y_cpu.numpy()
        predicted_labels_np = predicted_labels_cpu.numpy()
        macro_precision = precision_score(y_np, predicted_labels_np, average='macro')  
        print(f"acc: {acc}, f1: {f1}, macro_precision: {macro_precision}")
      
    def _fit_data_matrix_to_network_input(
    self, data_matrix: pd.DataFrame, feature, feature_column="Gene", 
        ) -> pd.DataFrame:
        nr_features_in_matrix = len(data_matrix.index)
        if len(feature) > nr_features_in_matrix:
            features_df = pd.DataFrame(feature, columns=[feature_column])
            data_matrix = data_matrix.merge(features_df, how="right", on=feature_column)
        if len(feature) > 0:
            data_matrix.set_index(feature_column, inplace=True)
            feature = list(feature)
      
            data_matrix = data_matrix.loc[feature]
        return data_matrix
    def _generate_k_folds(
        self,
        data_matrix: pd.DataFrame,
        design_matrix: pd.DataFrame,
        groups=list(range(1,25)),  
        n_folds=3,
        test_size=0.1, 
       
        random_state=42
    ):
        y = []
        dfs = []
        for i, group in enumerate(groups):
            group_samples = design_matrix[design_matrix["group"] == group]["sample"].values
            df_group = data_matrix[group_samples].T  
            dfs.append(df_group)
            y += [group-1 for _ in group_samples]    

        y = np.array(y)
        X = pd.concat(dfs).fillna(0).to_numpy()
    


        
        return X,y
    def calculate_accuracy(self, y, prediction) -> float:
        return torch.sum(y == prediction).item() / float(len(y))
    def calculate_F1(self, y, prediction) -> float:
        from torchmetrics.classification import MulticlassF1Score

        metric = MulticlassF1Score(num_classes=32).to("cpu")
        return metric(prediction, y)

