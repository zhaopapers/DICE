from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from binn import BINN
import numpy as np
import torch
import pandas as pd
import lightning.pytorch as pl
import networkx as nx
import dill
from torch import nn as nn
import pickle
from sklearn.utils.class_weight import compute_class_weight
class based_cell_train:
    def __init__(self, model: BINN, explainer, temperature_init=1.0, lr=0.001, residual=True):
        self.model = model
        #self.n_layers = model.n_layers
        self.learning_rate = model.learning_rate
        self.activation = model.activation
        self.activation_final = model.activation_final
        self.weight = model.weight
        self.explainer = explainer
  
        print(self.model)

    def fit(
        self,
        input_data,
        design_matrix,
        nr_iterations: int = 2,
        max_epochs: int = 1,
        n_folds= 3,
        test_size= 0.2,
        val_size = 0.2,
        connectivity_matrices_list: dict = None,
        batch_size = 16,
        early_stopping=True,
        gene_list = None,
        num_workers=8,
    ):
        if early_stopping:
            print(f"Will apply early stopping")
        return_dict = {
            "models": [],
            "train_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": [],
            "test_acc": [],
            "test_f1": [],
            "iteration": [],
            "epochs": [],
            "trainable_params": [],
            "matrices": [],
        }
       

        self.B_cell_connectivity_matrices = connectivity_matrices_list['B_cell_connectivity_matrices']
        self.CD4_cell_Tfh_connectivity_matrices = connectivity_matrices_list['CD4_cell_Tfh_connectivity_matrices']
        self.CD4_cell_Th1_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th1_connectivity_matrices']
        self.CD4_cell_Th17_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th17_connectivity_matrices']
        self.CD4_cell_Th2_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th2_connectivity_matrices']
        self.CD4_cell_Treg_connectivity_matrices = connectivity_matrices_list['CD4_cell_Treg_connectivity_matrices']
        self.CD8_cell_Tem_connectivity_matrices = connectivity_matrices_list['CD8_cell_Tem_connectivity_matrices']
        self.CD8_cell_Tcm_connectivity_matrices = connectivity_matrices_list['CD8_cell_Tcm_connectivity_matrices']
        intersection = gene_list
        for iteration in range(nr_iterations):
            print(f"---------------- Iteration: {iteration} ----------------")

            self.fitted_input_data ,feature= self._fit_data_matrix_to_network_input(
                input_data.reset_index(),
                features=intersection,
                feature_column="Gene"
            )
   
            
            
            splits = self._generate_k_folds(
                self.fitted_input_data, design_matrix=design_matrix,n_folds=n_folds, test_size= test_size, val_size=val_size
            )


            val_accs = []
            val_losses = []
            test_accs = []
    
            test_f1s = []
            train_accs = []
            train_losses = []
            epochs = []
            i=0
            for split in splits:
                X_train, y_train, X_val, y_val, X_test, y_test = split

            train_dataloader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(
                    torch.tensor(X_train, dtype=torch.float), torch.LongTensor(y_train)
                ),
                batch_size=batch_size,
                num_workers=0,
                shuffle=True,
            
                pin_memory=True
            )
            val_dataloader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(
                    torch.tensor(X_val, dtype=torch.float), torch.LongTensor(y_val)
                ),
                batch_size=batch_size,
                num_workers=0,
            
                pin_memory=True
            )
           
            test_dataloader = torch.utils.data.DataLoader(
                dataset=torch.utils.data.TensorDataset(
                    torch.tensor(X_test, dtype=torch.float), torch.LongTensor(y_test)
                ),
                batch_size=batch_size,
                num_workers=0,
                
                pin_memory=True
            )                
            callbacks = []
            if early_stopping:
                callbacks.append(
                       
                        pl.callbacks.early_stopping.EarlyStopping(
                            patience=10,
                            min_delta=0.001,
                            monitor="val_loss",
                            mode="min",
                        )
                        
                )
                
            print(np.unique(y_train))
            print(np.unique(y_val))     
            print(np.unique(y_test))  
            classes =  np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)

            self.model = BINN(
                    connectivity_matrices_list = connectivity_matrices_list,
                    #n_layers=self.n_layers,
                    dropout=0.2,
                    weight=class_weights,
                    validate=True,
                    residual=True,
                    activation=self.activation,
                    activation_final= self.activation_final,
                    scheduler="plateau",
                    learning_rate=self.learning_rate,
                    device="cuda:1"
                    
            )
            
            self.model = self.model.float()
            trainer = pl.Trainer(
                
                    max_epochs=max_epochs,
                    enable_progress_bar=False,
                    accelerator='gpu',
                    devices=[1],
                    
                    enable_model_summary=False,
                    callbacks=callbacks,
                    
            )
                
            trainer.fit(self.model, train_dataloader, val_dataloader)
            self.model.calibrate_temperature(val_dataloader)
            optimized_temperature = self.model.temperature.item()
            print(f"Optimized Temperature: {self.model.temperature.item():.4f}")
            train_accs.append(trainer.callback_metrics.get('train_acc'))
            train_losses.append(trainer.callback_metrics.get('train_loss'))
                
            val_dict = trainer.validate(self.model, val_dataloader)
                
            val_accs.append(val_dict[0]["val_acc"])
            val_losses.append(val_dict[0]["val_loss"])
            test_dict = trainer.test(self.model, test_dataloader)
            
            test_f1s.append(test_dict[0]["test_F1"])
            test_accs.append(test_dict[0]["test_acc"])


            epochs.append(self.model.current_epoch)
            self.test_data = torch.Tensor(np.concatenate([X_train, X_val, X_test], axis=0)).to("cuda:1")
            self.y = torch.Tensor(np.concatenate([y_train, y_val, y_test], axis=0)).to("cuda:1")
            self.background_data = torch.Tensor(
                np.concatenate([X_train, X_val, X_test], axis=0)
            ).to("cuda:1")

            torch.save(self.model, '/home/zxl/hdd/cellfate/model4/binn_model_unB{}_tpm_100_16.pth'.format(iteration))
                
            variables_to_save = { 'X_test': X_test, 'y_test': y_test, "features":feature}
            filename = '/home/zxl/hdd/cellfate/model4/test__unB{}_tpm_100_16.pkl'.format(iteration)
                # 创建 DataFrame
                
            with open(filename, 'wb') as f:
                dill.dump(variables_to_save, f)
          
            return_dict["test_acc"].append(test_accs)
            return_dict["test_f1"].append(test_f1s)
            return_dict["val_acc"].append(val_accs)
            return_dict["val_loss"].append(val_losses)
            return_dict["train_acc"].append(train_accs)
            return_dict["train_loss"].append(train_losses)
            return_dict["epochs"].append(epochs)
            return_dict["models"].append(self.model)
            return_dict["iteration"].append(iteration)
    
        return return_dict
        
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

        y = np.array(y)
        X = pd.concat(dfs).fillna(0).to_numpy()


        splits = []
        skf_outer = StratifiedShuffleSplit(n_splits=n_folds, test_size=test_size, random_state=random_state)
        
  
        val_size_relative = val_size / (1 - test_size)
        

        for train_val_index, test_index in skf_outer.split(X, y):
            X_train_val, X_test = X[train_val_index], X[test_index]
            y_train_val, y_test = y[train_val_index], y[test_index]
            

            skf_inner = StratifiedShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=random_state)
            for train_index, val_index in skf_inner.split(X_train_val, y_train_val):
                X_train, X_val = X_train_val[train_index], X_train_val[val_index]
                y_train, y_val = y_train_val[train_index], y_train_val[val_index]
                
 

                

  
                
                splits.append((X_train, y_train, X_val, y_val, X_test, y_test))
        
        return splits
    
    def fast_train(self, dataloader, num_epochs, optimizer):
        return_dict = {"accuracies":[], "losses":[], "epoch":[]}
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_accuracy = 0

            for _, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.to(self.model.device)
                targets = targets.to(self.model.device).type(torch.LongTensor)
                optimizer.zero_grad()
                outputs = self.model(inputs).to(self.model.device)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_accuracy += torch.sum(
                    torch.argmax(outputs, axis=1) == targets
                ) / len(targets)

            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)
            return_dict["accuracies"].append(avg_accuracy.numpy().tolist())
            return_dict["losses"].append(avg_loss)
            return_dict["epoch"].append(epoch)

        print(
            f"Final epoch: Average Accuracy {avg_accuracy:.5f}, Average Loss: {avg_loss:.2f}"
        )
        return self.model, return_dict
    
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
        return data_matrix ,features
