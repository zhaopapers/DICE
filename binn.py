import os

import collections
import torch
from lightning.pytorch import LightningModule
from torch import nn as nn
from torch.nn.utils import prune as prune
from .network import Network
from sklearn.metrics import precision_score


class BINN(LightningModule):
    """
    Implements a Biologically Informed Neural Network (BINN). The BINN
    is implemented using the Lightning-framework.
    If you are unfamiliar with PyTorch, we suggest visiting
    their website: https://pytorch.org/


    Args:
        pathways (Network): A Network object that defines the network topology.
        activation (str, optional): Activation function to use. Defaults to "tanh".
        weight (torch.Tensor, optional): Weights for loss function. Defaults to torch.Tensor([1, 1]).
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        n_layers (int, optional): Number of layers in the network. Defaults to 4.
        scheduler (str, optional): Learning rate scheduler to use. Defaults to "plateau".
        optimizer (str, optional): Optimizer to use. Defaults to "adam".
        validate (bool, optional): Whether to use validation data during training. Defaults to False.
        n_outputs (int, optional): Number of output nodes. Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.
        residual (bool, optional): Whether to use residual connections. Defaults to False.

    Attributes:
        residual (bool): Whether to use residual connections.
        pathways (Network): A Network object that defines the network topology.
        n_layers (int): Number of layers in the network.
        layer_names (List[str]): List of layer names.
        features (Index): A pandas Index object containing the input features.
        layers (nn.Module): The layers of the BINN.
        loss (nn.Module): The loss function used during training.
        learning_rate (float): Learning rate for optimizer.
        scheduler (str): Learning rate scheduler used.
        optimizer (str): Optimizer used.
        validate (bool): Whether to use validation data during training.
    """

    def __init__(
        self,
        network: Network = None,
     
        connectivity_matrices_list: dict = None,
        activation: str = "tanh",
        activation_final: str = "sigmoid",
        weight: torch.tensor = torch.ones(24),
        learning_rate: float = 1e-4,
        # n_layers: int = 5,
        scheduler: str = "plateau",
        optimizer: str = "adam",
        validate: bool = False,
        n_outputs: int = 24,
        dropout: float = 0,
        residual: bool = True,
        device: str = "cuda:0",
        temperature_init=1.0
    ):
        super().__init__()
     
        self.to(device)
        self.test_labels = []
        self.test_probs = []
        self.residual = residual
        self.batchnorm_np = nn.BatchNorm1d(7057, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True).to(self.device)

        # if not connectivity_matrices:
        #     self.network = network
        #     self.connectivity_matrices = self.network.get_connectivity_matrices(
        #         n_layers
        #     )
        # else:
        #     self.connectivity_matrices = connectivity_matrices
        # self.n_layers = n_layers

        self.temperature = nn.Parameter(torch.tensor(temperature_init), requires_grad=False)
        self.freeze()
        if  self.residual:

            self.B_cell_connectivity_matrices = connectivity_matrices_list['B_cell_connectivity_matrices']
            self.CD4_cell_Tfh_connectivity_matrices = connectivity_matrices_list['CD4_cell_Tfh_connectivity_matrices']
            self.CD4_cell_Th1_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th1_connectivity_matrices']
            self.CD4_cell_Th17_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th17_connectivity_matrices']
            self.CD4_cell_Th2_connectivity_matrices = connectivity_matrices_list['CD4_cell_Th2_connectivity_matrices']
            self.CD4_cell_Treg_connectivity_matrices = connectivity_matrices_list['CD4_cell_Treg_connectivity_matrices']
            self.CD8_cell_Tem_connectivity_matrices = connectivity_matrices_list['CD8_cell_Tem_connectivity_matrices']
            self.CD8_cell_Tcm_connectivity_matrices = connectivity_matrices_list['CD8_cell_Tcm_connectivity_matrices']
            def	layer_names_size(connectivity_matrices):		
                layer_sizes = []
                layer_names = []

                matrix = connectivity_matrices[0]
                i, _ = matrix.shape
                layer_sizes.append(i)
                layer_names.append(matrix.index.tolist())
                features = matrix.index
                trainable_params = matrix.to_numpy().sum() + len(matrix.index)
                for matrix in connectivity_matrices[1:]:
                    trainable_params += matrix.to_numpy().sum() + len(matrix.index)
                    i, _ = matrix.shape
                    layer_sizes.append(i)
                    layer_names.append(matrix.index.tolist())
                return(layer_sizes, layer_names)
            
            self.B_cell_layer_sizes, self.B_cell_layer_names = layer_names_size(self.B_cell_connectivity_matrices)
            self.CD8_cell_Tcm_layer_sizes, self.CD8_cell_Tcm_layer_names = layer_names_size(self.CD8_cell_Tcm_connectivity_matrices)
            self.CD8_cell_Tem_layer_sizes, self.CD8_cell_Tem_layer_names = layer_names_size(self.CD8_cell_Tem_connectivity_matrices)
            self.CD4_cell_Th1_layer_sizes, self.CD4_cell_Th1_layer_names = layer_names_size(self.CD4_cell_Th1_connectivity_matrices)
            self.CD4_cell_Th2_layer_sizes, self.CD4_cell_Th2_layer_names = layer_names_size(self.CD4_cell_Th2_connectivity_matrices)
            self.CD4_cell_Th17_layer_sizes, self.CD4_cell_Th17_layer_names = layer_names_size(self.CD4_cell_Th17_connectivity_matrices)
            self.CD4_cell_Tfh_layer_sizes, self.CD4_cell_Tfh_layer_names = layer_names_size(self.CD4_cell_Tfh_connectivity_matrices)
            self.CD4_cell_Treg_layer_sizes, self.CD4_cell_Treg_layer_names = layer_names_size(self.CD4_cell_Treg_connectivity_matrices)
            
            self.B_cell_layers = _generate_residual_deep(
                self.B_cell_layer_sizes,
                connectivity_matrices=self.B_cell_connectivity_matrices,
                activation=activation,
             
                LSO=[3,9,15,21,27,33],   LSG=[5,11,17,23,29],
                bias=True
            ).to(self.device)		
            self.CD8_cell_Tcm_layers = _generate_residual_deep(
                self.CD8_cell_Tcm_layer_sizes,
                connectivity_matrices=self.CD8_cell_Tcm_connectivity_matrices,
                activation=activation,
             
                LSO=[3,9,15,21,27,33],   LSG=[5,11,17,23,29],
                bias=True
            ).to(self.device)
            self.CD8_cell_Tem_layers = _generate_residual_deep(
                self.CD8_cell_Tem_layer_sizes,
                connectivity_matrices=self.CD8_cell_Tem_connectivity_matrices,
                activation=activation,
                
                LSO=[3,9,15,21,27,33],   LSG=[5,11,17,23,29],
                bias=True
            ).to(self.device)
            self.CD4_cell_Th1_layers =  _generate_residual_deep(
                self.CD4_cell_Th1_layer_sizes,
                connectivity_matrices=self.CD4_cell_Th1_connectivity_matrices,
                activation=activation,
               
                LSO=[3,9,15,21,27],   LSG=[5,11,17,23],
                bias=True
            ).to(self.device)
            self.CD4_cell_Th2_layers = _generate_residual_deep(
                self.CD4_cell_Th2_layer_sizes,
                connectivity_matrices=self.CD4_cell_Th2_connectivity_matrices,
                activation=activation,
               
                LSO=[3,9,15,21,27],   LSG=[5,11,17,23],
                bias=True
            ).to(self.device)
            self.CD4_cell_Th17_layers = _generate_residual_deep(
                self.CD4_cell_Th17_layer_sizes,
                connectivity_matrices=self.CD4_cell_Th17_connectivity_matrices,
                activation=activation,
                
                LSO=[3,9,15,21,27],   LSG=[5,11,17,23],
                bias=True
            ).to(self.device)
            self.CD4_cell_Tfh_layers = _generate_residual_deep(
                self.CD4_cell_Tfh_layer_sizes,
                connectivity_matrices=self.CD4_cell_Tfh_connectivity_matrices,
                activation=activation,
            
                LSO=[3,9,15,21,27],   LSG=[5,11,17,23],
                bias=True
            ).to(self.device)
            self.CD4_cell_Treg_layers = _generate_residual_deep(
                self.CD4_cell_Treg_layer_sizes,
                connectivity_matrices=self.CD4_cell_Treg_connectivity_matrices,
                activation=activation,
               
                LSO=[3,9,15,21,27],   LSG=[5,11,17,23],
                bias=True
             
            ).to(self.device)
         
            self.final_layers= _generate_final(688,  bias=False, n_outputs = 24).to(self.device)	
        else:
            layer_sizes = []
            self.layer_names = []
            matrix = self.connectivity_matrices[0]
            i, _ = matrix.shape
            layer_sizes.append(i)
            self.layer_names.append(matrix.index.tolist())
            self.features = matrix.index
            self.trainable_params = matrix.to_numpy().sum() + len(matrix.index)
            for matrix in self.connectivity_matrices[1:]:
                self.trainable_params += matrix.to_numpy().sum() + len(matrix.index)
                i, _ = matrix.shape
                layer_sizes.append(i)
                self.layer_names.append(matrix.index.tolist())
            self.layers = _generate_sequential(
                layer_sizes,
                connectivity_matrices=self.connectivity_matrices,
                activation=activation,
                bias=True,
                n_outputs=n_outputs,
                dropout=dropout,
                )
        self.apply(_init_weights)
        self.weight = weight
        self.loss = nn.CrossEntropyLoss(weight=torch.tensor(self.weight, device=device))
        self.learning_rate = learning_rate
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.validate = validate
        self.activation = activation
        self.activation_final = activation_final 
        self.save_hyperparameters()
        print("\nBINN is on the device:", self.device, end="\n")
        
    def forward(self, x: torch.tensor, use_temperature: bool = False) -> torch.tensor:
        """
        Performs a forward pass through the BINN.

        Args:
            x (torch.Tensor): The input tensor to the BINN.

        Returns:
            torch.Tensor: The output tensor of the BINN.
        """
        if self.residual:
            outputs = []
            self.cell_layers = self.B_cell_layers
            outputs.append(self._forward_residual(x))
            self.cell_layers = self.CD8_cell_Tcm_layers
            outputs.append(self._forward_residual(x))  
            self.cell_layers = self.CD8_cell_Tem_layers
            outputs.append(self._forward_residual(x))       
            self.cell_layers = self.CD4_cell_Th1_layers
            outputs.append(self._forward_residual(x))       
            self.cell_layers = self.CD4_cell_Th2_layers
            outputs.append(self._forward_residual(x))      
            self.cell_layers = self.CD4_cell_Th17_layers
            outputs.append(self._forward_residual(x))  
            self.cell_layers = self.CD4_cell_Tfh_layers
            outputs.append(self._forward_residual(x))
            self.cell_layers = self.CD4_cell_Treg_layers
            outputs.append(self._forward_residual(x))
            combined = torch.cat(outputs, dim=1)  
        
        
            final_output = self._forward_final(combined)
            return final_output
        
        else:
            return self.layers(x)
        if use_temperature:
            return final_output / self.temperature
        else:
            return final_output

    def training_step(self, batch, _):
        """
        Performs a single training step for the BINN.

        Args:
            batch: The batch of data to use for the training step.
            _: Not used.

        Returns:
            torch.Tensor: The loss tensor for the training step.
        """
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x, use_temperature=False).to(self.device)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        #print(f"Training - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    def validation_step(self, batch, _):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self(x, use_temperature=True)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        #print(f"Validation - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": accuracy}


    

    def test_step(self, batch, _):
        """
        Implements a single testing step for the BINN.

        Args:
            batch: A tuple containing the input and output data for the current batch.
            _: The batch index, which is not used.
        """
        x, y = batch
        x = x.to("cuda:1")
        y = y.to("cuda:1")
        y_hat = self(x, use_temperature=True)
        self.test_labels.append(y)
        self.test_probs.append(y_hat)



    def on_test_epoch_end(self):
  
        test_labels = torch.cat(self.test_labels)
        test_probs = torch.cat(self.test_probs)
    
        prediction = torch.argmax(test_probs, dim=1)
        self.test_labels.clear()
        self.test_probs.clear()


        accuracy = self.calculate_accuracy(test_labels, prediction)

        F1 = self.calculate_F1(test_labels, prediction)

        

        self.log("test_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_F1", F1, prog_bar=True, on_step=False, on_epoch=True)

        return { "test_acc": accuracy, "test_F1": F1}

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training the BINN.

        Returns:
            A list of optimizers and a list of learning rate schedulers.
        """
        if self.validate:
            monitor = "val_loss"
        else:
            monitor = "train_loss"

        if isinstance(self.optimizer, str):
            if self.optimizer == "adam":
                optimizer = torch.optim.Adam([
                    {'params': self.B_cell_layers.parameters()},
                    {'params': self.CD8_cell_Tcm_layers.parameters()},
                    {'params': self.CD8_cell_Tem_layers.parameters()},
                    {'params': self.CD4_cell_Th1_layers.parameters()},
                    {'params': self.CD4_cell_Th2_layers.parameters()},
                    {'params': self.CD4_cell_Th17_layers.parameters()},
                    {'params': self.CD4_cell_Tfh_layers.parameters()},
                    {'params': self.CD4_cell_Treg_layers.parameters()},
                    {'params': self.final_layers.parameters()}], lr=self.learning_rate, weight_decay=1e-4)
                self.optimizer = optimizer
        else:
            optimizer = self.optimizer

        if self.scheduler == "plateau":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, threshold=0.01, mode="min"
                ),
                "interval": "epoch",
                "monitor": monitor,
            }
        elif self.scheduler == "step":
            scheduler = {
                "scheduler": torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=25, gamma=0.1
                )
            }

        return [optimizer], [scheduler]

    def calculate_accuracy(self, y, prediction) -> float:
        return torch.sum(y == prediction).item() / float(len(y))
    def calculate_F1(self, y, prediction) -> float:
        from torchmetrics.classification import MulticlassF1Score

        metric = MulticlassF1Score(num_classes=24).to("cuda:1")
        return metric(prediction, y)
    def calculate_AUC(self, y, positive_class_probabilities) -> float:
        from sklearn.metrics import roc_auc_score
        import numpy as np
        y_np = y.cpu().numpy()
        prediction_np = positive_class_probabilities.cpu().numpy()

        if len(np.unique(y_np)) < 2:
 
            return 0.0  
        auc = roc_auc_score(y_np, prediction_np)
        return auc
    def get_connectivity_matrices(self) -> list:
        """
        Returns the connectivity matrices underlying the BINN.

        Returns:
            The connectivity matrices as a list of Pandas DataFrames.
        """
        return self.connectivity_matrices

    def reset_params(self):
        """
        Resets the trainable parameters of the BINN.
        """
        self.apply(_reset_params)

    def init_weights(self):
        """
        Initializes the trainable parameters of the BINN.
        """
        self.apply(_init_weights)

    def _forward_residual(self, x: torch.tensor):
        
        
        x_final = torch.tensor([], device=self.device)
        x =x.to(self.device)        

        Gene_input= self.batchnorm_np(x)
        for name, layer in self.cell_layers.named_children():
			
            if name.startswith("differentiation"):
                
                x = layer(x)
                x_final = torch.cat((x_final, x), dim=1)

            elif name.startswith("Gene_in"): 
                
                x = layer(x)
                x = self.batchnorm_np(x)
                
                x=(Gene_input+x)/2
            else:
                x = layer(x)
				
        return x_final

    def _forward_final(self, x: torch.tensor):
        x = x.to(self.device)
        
        self.final_layers.to(self.device)
        for name, layer in self.final_layers.named_children():
        
            x = layer(x)
        
        return x
    def calibrate_temperature(self, val_loader):
          
        self.temperature.requires_grad = True
    
        for param in self.parameters():
            param.requires_grad = False
        self.temperature.requires_grad = True
            
       
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)
 
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in val_loader:
                logits = self(x, use_temperature=False)  
                logits_list.append(logits)
                labels_list.append(y)
        logits_val = torch.cat(logits_list)
        labels_val = torch.cat(labels_list)
            
            
        def closure():
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(logits_val / self.temperature, labels_val)
            loss.backward()
            return loss
            
        for _ in range(100): 
            optimizer.step(closure)
         
        for param in self.parameters():
            param.requires_grad = True
        self.temperature.requires_grad = False


def _init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def _reset_params(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def _append_activation(layers, activation, n):
    if activation == "tanh":
        layers.append((f"Tanh {n}", nn.Tanh()))
    elif activation == "relu":
        layers.append((f"ReLU {n}", nn.ReLU()))
    elif activation == "leaky relu":
        layers.append((f"LeakyReLU {n}", nn.LeakyReLU()))
    elif activation == "sigmoid":
        layers.append((f"Sigmoid {n}", nn.Sigmoid()))
    elif activation == "elu":
        layers.append((f"Elu {n}", nn.ELU()))
    elif activation == "hardsigmoid":
        layers.append((f"HardSigmoid {n}", nn.Hardsigmoid()))
    return layers


def _generate_sequential(
    layer_sizes,
    connectivity_matrices=None,
    activation: str = "tanh",
    bias: bool = True,
    n_outputs: int = 2,
    dropout: int = 0,
):
    layers = []
    for n in range(len(layer_sizes) - 1):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))  # linear layer
        layers.append((f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n + 1])))
        if connectivity_matrices is not None:
            prune.custom_from_mask(
                linear_layer,
                name="weight",
                mask=torch.tensor(connectivity_matrices[n].T.values),
            )
        if isinstance(dropout, list):
            layers.append((f"Dropout_{n}", nn.Dropout(dropout[n])))
        else:
            layers.append((f"Dropout_{n}", nn.Dropout(dropout)))
        if isinstance(activation, list):
            layers.append((f"Activation_{n}", activation[n]))
        else:
            _append_activation(layers, activation, n)
    layers.append(("Output layer", nn.Linear(layer_sizes[-1], n_outputs, bias=bias)))
    model = nn.Sequential(collections.OrderedDict(layers))
    return model



def _generate_residual_deep(
    layer_sizes, connectivity_matrices=None, activation="tanh",LSO=[3,9,15,21,27,33],   LSG=[5,11,17,23,29], bias=False, n_outputs=2
):
    layers = []

    def generate_block(n, layers, activation):
        linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
        layers.append((f"Layer_{n}", linear_layer))
        layers.append((f"BatchNorm_{n}", nn.BatchNorm1d(layer_sizes[n + 1])))
        if connectivity_matrices is not None:
            prune.custom_from_mask(
                linear_layer,
                name="weight",
                mask=torch.tensor(connectivity_matrices[n].T.values),
            )
        layers.append((f"Dropout_{n}", nn.Dropout(0.2)))
        if activation == "tanh":
            layers.append((f"Tanh {n}", nn.Tanh()))
        elif activation == "relu":
            layers.append((f"ReLU {n}", nn.ReLU()))
        elif activation == "leaky relu":
            layers.append((f"LeakyReLU {n}", nn.LeakyReLU()))
        elif activation == "sigmoid":
            layers.append((f"Sigmoid {n}", nn.Sigmoid()))
        elif activation == "elu":
            layers.append((f"Elu {n}", nn.ELU()))
        elif activation == "hardsigmoid":
            layers.append((f"HardSigmoid {n}", nn.Hardsigmoid()))
        return layers

    for n in range(len(layer_sizes)-1):
        if n in LSO:
            linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
            layers.append((f"Layer_{n}", linear_layer))
            layers.append(("BatchNorm_"+str(n), nn.BatchNorm1d(layer_sizes[n + 1])))
            layers.append(("Dropout_"+str(n), nn.Dropout(0.2)))
            if connectivity_matrices is not None:
                prune.custom_from_mask(
                linear_layer,
                name="weight",
                mask=torch.tensor(connectivity_matrices[n].T.values),
            )
            if activation == "tanh":
                layers.append((f"differentiation_Tanh {n}", nn.Tanh()))
            elif activation == "relu":
                layers.append((f"differentiation_ReLU {n}", nn.ReLU()))
            elif activation == "leaky relu":
                layers.append((f"differentiation_LeakyReLU {n}", nn.LeakyReLU()))
            elif activation == "sigmoid":
                layers.append((f"differentiation_Sigmoid {n}", nn.Sigmoid()))
            elif activation == "elu":
                layers.append((f"differentiation_Elu {n}", nn.ELU()))
            elif activation == "hardsigmoid":
                layers.append((f"differentiation_HardSigmoid {n}", nn.Hardsigmoid()))
        
        elif n in LSG:
            linear_layer = nn.Linear(layer_sizes[n], layer_sizes[n + 1], bias=bias)
            layers.append((f"Gene_in_Layer_{n}", linear_layer))
            layers.append(("Dropout_"+str(n), nn.Dropout(0.2)))
            if connectivity_matrices is not None:
                prune.custom_from_mask(
                linear_layer,
                name="weight",
                mask=torch.tensor(connectivity_matrices[n].T.values),
            )
            if activation == "tanh":
                layers.append((f"Tanh {n}", nn.Tanh()))
            elif activation == "relu":
                layers.append((f"ReLU {n}", nn.ReLU()))
            elif activation == "leaky relu":
                layers.append((f"LeakyReLU {n}", nn.LeakyReLU()))
            elif activation == "sigmoid":
                layers.append((f"Sigmoid {n}", nn.Sigmoid()))
            elif activation == "elu":
                layers.append((f"Elu {n}", nn.ELU()))
            elif activation == "hardsigmoid":
                layers.append((f"HardSigmoid {n}", nn.Hardsigmoid()))
        else:
            layers = generate_block(n, layers, activation)
           

   

    model = nn.Sequential(collections.OrderedDict(layers))


    return(model)
	

	


def _generate_final(final_cat: int , bias=False, n_outputs = 24):
    layers = []   

    layers.append(("BatchNorm_{n}", nn.BatchNorm1d(final_cat)))
    layers.append(("Dropout", nn.Dropout(0.2)))
    layers.append(
            (
                "final",
                nn.Linear(final_cat, n_outputs, bias=bias),
            )
        )
    

    layers.append(("identity_final",nn.Identity()))
    model = nn.Sequential(collections.OrderedDict(layers))

    return(model)





