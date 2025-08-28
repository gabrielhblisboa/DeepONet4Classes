import torch
import torch.nn as nn
import functools
import typing

from src.models.base_model import BaseModel

# class DeepONet(BaseModel):
#     """
#     Implementação da Deep Operator Network (DeepONet) para classificação.
#     Esta versão é flexível e aceita qualquer módulo PyTorch como Branch Net.
#     """
#     def __init__(self,
#                  branch_net: torch.nn.Module,
#                  n_targets: int,
#                  embedding_dim: int = 128,
#                  use_bias: bool = True):
#         super().__init__()

#         self.branch_net = branch_net

#         self.trunk_net = torch.nn.Embedding(num_embeddings=n_targets, embedding_dim=embedding_dim)

#         self.use_bias = use_bias
#         if self.use_bias:
#             self.bias = torch.nn.Parameter(torch.zeros(embedding_dim))


#     def forward(self, data: torch.Tensor) -> torch.Tensor:
#         # branch_output -> [batch_size, embedding_dim]
#         branch_output = self.branch_net(data)

#         # trunk_prototypes -> [n_targets, embedding_dim]
#         trunk_prototypes = self.trunk_net.weight

#         # Produto escalar via multiplicação de matrizes
#         logits = torch.matmul(branch_output, trunk_prototypes.t())

#         if self.use_bias:
#             logits = logits + self.bias
            
#         return logits

class DeepONet(BaseModel):
    """
    Implementação da DeepONet adaptada para classificação multiclasse direta.
    """
    def __init__(self, branch_net: nn.Module, trunk_net: nn.Module, num_classes: int = 4):
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.num_classes = num_classes
        
        # O bias agora é um vetor, um para cada classe
        self.b = torch.nn.Parameter(torch.randn(self.num_classes))
        
        # Cria a matriz identidade (one-hots) que será o input constante da Trunk Net
        # O register_buffer garante que este tensor seja movido para a GPU junto com o modelo
        self.register_buffer('identity_matrix', torch.eye(self.num_classes))

    def forward(self, x_branch):
        # 1. Branch Net processa o lote de sinais
        #    Saída: (batch_size, p_dim)
        branch_out = self.branch_net(x_branch)
        
        # 2. Trunk Net processa a matriz identidade para obter os "moldes" de todas as classes
        #    Saída: (num_classes, p_dim)
        trunk_out = self.trunk_net(self.identity_matrix)
        
        # 3. Multiplicação de matrizes para obter os logits
        #    (batch_size, p_dim) @ (p_dim, num_classes) -> (batch_size, num_classes)
        logits = torch.matmul(branch_out, trunk_out.t()) + self.b
        
        return logits

