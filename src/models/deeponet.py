import torch
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
                 n_targets: int,
                 embedding_dim: int = 128,
                 use_bias: bool = True):
        super().__init__()

        self.branch_net = branch_net

        self.trunk_net = torch.nn.Embedding(num_embeddings=n_targets, embedding_dim=embedding_dim)

        self.use_bias = use_bias
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(embedding_dim))


    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # branch_output -> [batch_size, embedding_dim]
        branch_output = self.branch_net(data)

        # trunk_prototypes -> [n_targets, embedding_dim]
        trunk_prototypes = self.trunk_net.weight

        # Produto escalar via multiplicação de matrizes
        logits = torch.matmul(branch_output, trunk_prototypes.t())

        if self.use_bias:
            logits = logits + self.bias
            
        return logits