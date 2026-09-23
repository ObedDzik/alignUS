from torch.nn.functional import relu, dropout
import torch
from torch import nn
import typing as tp


class MLPv1(nn.Module): 
    def __init__(self, *dims: tp.Sequence[int]):
        super().__init__()
        self.layers = nn.ModuleList() 

        for i in range(len(dims) - 1): 
            last_layer = i + 1 == len(dims) - 1

            self.layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=dims[i], 
                        out_features=dims[i+1]
                    ), 
                    nn.ReLU() if not last_layer else nn.Identity()
                )
            )

    def forward(self, X): 
        for layer in self.layers: 
            X = layer(X)

        return X 


class MLP(torch.nn.Module):
    def __init__(self, *inner_dims, dropout=0, use_batchnorm=False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(inner_dims) - 1):
            self.layers.append(
                torch.nn.Linear(
                    in_features=inner_dims[i], out_features=inner_dims[i + 1]
                )
            )
            if use_batchnorm:
                self.layers.append(torch.nn.BatchNorm1d(inner_dims[i + 1]))
            self.layers.append(torch.nn.Dropout(p=dropout))
            self.layers.append(torch.nn.ReLU())

        self.dropout = dropout

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X


def MLPClassifier(*inner_dims, dropout=0.1, num_classes=2):
    return torch.nn.Sequential(
        MLP(*inner_dims, dropout=dropout), torch.nn.Linear(inner_dims[-1], num_classes)
    )


def MLPRegressor(*inner_dims, dropout=0.1):
    return torch.nn.Sequential(
        MLP(*inner_dims, dropout=dropout), torch.nn.Linear(inner_dims[-1], 1)
    )
