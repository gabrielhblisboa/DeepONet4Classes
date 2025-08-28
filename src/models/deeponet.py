import torch
import torch.nn as nn
import functools
import typing

from src.models.base_model import BaseModel

class DeepONet(BaseModel):
    """
    Implementação da Deep Operator Network (DeepONet) para classificação.
    Esta versão é flexível e aceita qualquer módulo PyTorch como Branch Net.
    """
    def __init__(self,
                 branch_net: torch.nn.Module,
                 trunk_net: torch.nn.Module,
                 use_bias: bool = True):
        super().__init__()

        self.branch_net = branch_net

        self.trunk_net = trunk_net
        self.use_bias = use_bias
        
        self.class_head = nn.Linear(in_features= 32, out_features=4)
        
        if use_bias:
            self.bias = torch.nn.Parameter(torch.randn(1))


    def forward(self, data: torch.Tensor, coords: torch.Tensor,) -> torch.Tensor:
        # branch_output -> [batch_size, embedding_dim]
        branch_output = self.branch_net(data)
        
        trunk_output = self.trunk_net(coords)
        
        
        print(f'------- branch shape ------->{branch_output.shape}')
        print(f'------- trunk shape ------->{trunk_output.shape}')
        # Produto escalar via multiplicação de matrizes
        logits = torch.matmul(branch_output, trunk_output.t())
        # logits = torch.einsum("bf,bf->bf", branch_output, trunk_output)

        if self.use_bias:
            logits = logits + self.bias
        # [32,4][32,4]
        print('------- logit shape ------->')
        print(logits.shape)
        
        y_pred = self.class_head(logits)
        return y_pred