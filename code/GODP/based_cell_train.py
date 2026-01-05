from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from binn import BINN
import numpy as np
import torch
import pandas as pd
import lightning.pytorch as pl
import networkx as nx
import dill

class based_cell_train:
    def __init__(self, model: BINN, explainer):
        self.model = model
        self.n_layers = model.n_layers
        self.learning_rate = model.learning_rate
        self.activation = model.activation
        self.activation_final = model.activation_final
        self.weight = model.weight
        self.explainer = explainer
        print(self.activation_final)
        self.connectivity_matrices = model.connectivity_matrices

    def fit(
        self,
        input_data,
        design_matrix,
        nr_iterations: int = 2,
        max_epochs: int = 1,
        n_folds= 3,
        batch_size = 16,
        early_stopping=True,
        num_workers= 10,
        dir=""
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
            "test_auc": [],
            "iteration": [],
            "epochs": [],
            "trainable_params": [],
            "matrices": [],
        }
        dfs = {}
        for iteration in range(nr_iterations):
            print(f"---------------- Iteration: {iteration} ----------------")

            self.fitted_input_data = self._fit_data_matrix_to_network_input(
                input_data.reset_index(),
                features=self.connectivity_matrices[0].index.values.tolist(),
                feature_column="Gene"
            )

            splits = self._generate_k_folds(
                self.fitted_input_data, design_matrix=design_matrix,n_folds=n_folds
            )


            val_accs = []
            val_losses = []
            test_accs = []
            test_aucs = []
            test_f1s = []
            train_accs = []
            train_losses = []
            epochs = []
            i=0
            for split in splits:
                
                self.model = BINN(
                    connectivity_matrices=self.connectivity_matrices,
                    n_layers=self.n_layers,
                    dropout=0.2,
                    weight=self.weight,
                    validate=True,
                    residual=False,
                    activation=self.activation,
                    activation_final= self.activation_final,
                    scheduler="plateau",
                    learning_rate=self.learning_rate,
                    device="cuda:1"
                    
                )
                X_train, y_train, X_val, y_val, X_test, y_test = split
                train_dataloader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.TensorDataset(
                        torch.Tensor(X_train), torch.LongTensor(y_train)
                    ),
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=True,
                )
                val_dataloader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.TensorDataset(
                        torch.Tensor(X_val), torch.LongTensor(y_val)
                    ),
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
           
                test_dataloader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.TensorDataset(
                        torch.Tensor(X_test), torch.LongTensor(y_test)
                    ),
                    batch_size=batch_size,
                    num_workers=0,
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

                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    enable_progress_bar=False,
                    accelerator='gpu',
                    devices=[1],
                    enable_model_summary=False,
                    callbacks=callbacks,
                    
                )
                print(self.model)
                trainer.fit(self.model, train_dataloader, val_dataloader)
                train_accs.append(trainer.callback_metrics.get('train_acc'))
                train_losses.append(trainer.callback_metrics.get('train_loss'))
           
                val_dict = trainer.validate(self.model, val_dataloader)

                val_accs.append(val_dict[0]["val_acc"])
                val_losses.append(val_dict[0]["val_loss"])
                test_dict = trainer.test(self.model, test_dataloader)
                test_aucs.append(test_dict[0]["test_AUC"])
                test_f1s.append(test_dict[0]["test_F1"])
                test_accs.append(test_dict[0]["test_acc"])


                epochs.append(self.model.current_epoch)

                self.test_data = torch.Tensor(np.concatenate([X_train, X_val, X_test], axis=0))
                self.background_data = torch.Tensor(
                    np.concatenate([X_train, X_val, X_test], axis=0)
                )
                self.explainer.update_model(self.model)
                variables_to_save = { 'test_data': self.test_data}
                filename = '/home/zxl/hdd/lsy/model_Th2/variables{}_fold{}.pkl'.format(iteration, i)
                print(filename)
                with open(filename, 'wb') as f:
                    dill.dump(variables_to_save, f)
                shap_dict = self.explainer._explain_layers(
                    self.test_data, self.background_data
                )

            
                feature_dict = {
                    "source": [],
                    "target": [],
                    "source name": [],
                    "target name": [],
                    "value": [],
                    "type": [],
                    "source layer":[],
                    "target layer": [],
                    "spilt": []
                }
                connectivity_matrices = self.model.get_connectivity_matrices()
                feature_id_mapping = {}
                feature_id = 0
                feature_id_mapping["root"] = feature_id
                for layer_features in shap_dict["features"]:
                    for feature in layer_features:
                        feature_id += 1
                        feature_id_mapping[feature] = feature_id

                curr_layer = 0

                for sv, features, cm in zip(
                    shap_dict["shap_values"], shap_dict["features"], connectivity_matrices
                ):
                    sv = np.asarray(sv)
                    sv = abs(sv)
                    sv_mean = np.mean(sv, axis=0)
         
                    for feature in range(sv_mean.shape[0]):
               
                        n_classes = sv_mean.shape[1]
                        connections = cm[cm.index == features[feature]]
                        connections = connections.loc[
                            :, (connections != 0).any(axis=0)
                        ]  # get targets and append to target
                        for target in connections:
                            for curr_class in range(n_classes):
                                feature_dict["source"].append(
                                    feature_id_mapping[features[feature]]
                                )
                                
                                feature_dict["target"].append(feature_id_mapping[target])
                                feature_dict["source name"].append(features[feature])
                                feature_dict["target name"].append(target)
                                feature_dict["value"].append(sv_mean[feature][curr_class])
                                feature_dict["type"].append(curr_class)
                                feature_dict["source layer"].append(curr_layer)
                                feature_dict["target layer"].append(curr_layer + 1)
                                feature_dict["spilt"].append(i)
                    curr_layer += 1
                df = pd.DataFrame(data=feature_dict)
                filename = "{}model_inter{}_fold{}.pth".format(dir,iteration, i)
                torch.save(self.model, filename)
                inaa = (n_folds*iteration)+i
                dfs[inaa] = df
                i=i+1

            return_dict["test_acc"].append(test_accs)

            return_dict["test_auc"].append(test_aucs)
            return_dict["test_f1"].append(test_f1s)
            return_dict["val_acc"].append(val_accs)
            return_dict["val_loss"].append(val_losses)
            return_dict["train_acc"].append(train_accs)
            return_dict["train_loss"].append(train_losses)
            return_dict["epochs"].append(epochs)
            return_dict["trainable_params"].append(self.model.trainable_params)
            return_dict["models"].append(self.model)
            return_dict["iteration"].append(iteration)
            '''    # 保存 df 到 CSV 文件，追加模式
            output_file = 'CLP_proB_output_file.csv'
            if os.path.exists(output_file):
                df.to_csv(output_file, mode='a', index=False, header=False)
            else:
                df.to_csv(output_file, index=False)

                # 保存 return_dict 到 CSV 文件，追加模式
            return_dict_df = pd.DataFrame.from_dict(return_dict, orient='index')
            return_dict_file = 'CLP_proB_return_dict.csv'
            if os.path.exists(return_dict_file):
                return_dict_df.to_csv(return_dict_file, mode='a', index=False, header=False)
            else:
                    return_dict_df.to_csv(return_dict_file, index=False)'''
            


        
        col_names = [f"value_{n}" for n in range(len(list(dfs.keys())))]
        values = [df.value.values for df in dfs.values()]
        values = np.array(values)
        values_mean = np.mean(values, axis=0)
        values_std = np.std(values, axis=0)
        df = dfs[0].copy()
        df.drop(columns=["value"], inplace=True)
        df[col_names] = values.T
        df["value_mean"] = values_mean
        df["values_std"] = values_std
        df["value"] = values_mean
        return df,return_dict
        
    def _generate_k_folds(
        self,
        data_matrix: pd.DataFrame,
        design_matrix: pd.DataFrame,
        groups=[1, 2],
        n_folds=3,
    ):
        y = []
        dfs = []
        for i, group in enumerate(groups):
            group_columns = design_matrix[design_matrix["group"] == group][
                "sample"
            ].values
            df = data_matrix[group_columns].T
            dfs.append(df)
            y += [i for _ in group_columns]
        y = np.array(y)
        X = pd.concat(dfs).fillna(0).to_numpy()
        X = preprocessing.StandardScaler().fit_transform(X)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        skf_test = StratifiedKFold(n_splits=10, shuffle=True)
        train_val_index, test_index = next(skf_test.split(X, y))
        splits = []
        X_train_val = X[train_val_index, :]
        y_train_val = y[train_val_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        for train_index, val_index in skf.split(X_train_val, y_train_val):
            X_train = X[train_index, :]
            y_train = y[train_index]
            X_val = X[val_index, :]
            y_val = y[val_index]
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
            data_matrix = data_matrix.loc[features]
        return data_matrix
