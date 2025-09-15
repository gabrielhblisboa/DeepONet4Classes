# import torch
# from torch import nn
# import torch.nn.functional as F
# from src.models.kan.KANConv import KAN_Convolutional_Layer

# class CKAN(nn.Module):
#     def __init__(self, input_shape, grid_size: int = 5, dropout_rate: float = 0.5, hidden_channels: list = [128], n_targets: int = 4, **kwargs):
#         super().__init__()
        
#         # Passando o argumento 'padding=1' diretamente para a camada
#         self.conv1 = KAN_Convolutional_Layer(in_channels=1, out_channels=16, kernel_size=3, grid_size=grid_size, padding=1)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
#         self.conv2 = KAN_Convolutional_Layer(in_channels=16, out_channels=32, kernel_size=3, grid_size=grid_size, padding=1)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.flat = nn.Flatten()
        
#         # Dummy forward pass para calcular o tamanho do flatten
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, *input_shape)
#             dummy_output = self._forward_features(dummy_input)
#             flattened_size = dummy_output.shape[1]
        
#         # Cabeça de classificação
#         self.class_head = nn.Sequential(
#             nn.Linear(flattened_size, hidden_channels[0]),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_channels[0], n_targets)
#         )
        
#         self.name = f"CKAN_(gs={grid_size})"

#     def _forward_features(self, x):
#         # O padding agora é tratado dentro da camada KANConv, não precisamos mais do F.pad aqui
#         x = self.pool1(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool2(F.relu(self.bn2(self.conv2(x))))
#         x = self.flat(x)
#         return x

#     def forward(self, x):
#         x = self._forward_features(x)
#         x = self.class_head(x)
#         return x

#     def get_embedding(self, x):
#         features = self._forward_features(x)
#         embedding = features
#         for layer in self.class_head[:-1]:
#             embedding = layer(embedding)
#         return embedding

import torch
from torch import nn
import torch.nn.functional as F
from src.models.kan.KANConv import KAN_Convolutional_Layer

class CKAN(nn.Module):
    def __init__(self, input_shape, grid_size: int = 5, dropout_rate: float = 0.5, hidden_channels: list = [128], n_targets: int = 4, latent_dim_size: int = 128, **kwargs):
        super().__init__()
        
        # Encoder com arquitetura baseada na sua CNN
        self.encoder = nn.Sequential(
            # Camada 1: 1 -> 16
            KAN_Convolutional_Layer(in_channels=1, out_channels=16, kernel_size=5, stride=2, padding=2, grid_size=grid_size),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),

            # Camada 2: 16 -> 32
            KAN_Convolutional_Layer(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2, grid_size=grid_size),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),

            # Camada 3: 32 -> 64
            KAN_Convolutional_Layer(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2, grid_size=grid_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),

            # Camada 4: 64 -> 128 (latent_dim_size)
            KAN_Convolutional_Layer(in_channels=64, out_channels=latent_dim_size, kernel_size=5, stride=2, padding=2, grid_size=grid_size),
            nn.BatchNorm2d(latent_dim_size),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )

        self.flat = nn.Flatten()
        
        # Dummy forward pass para calcular o tamanho do flatten
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.encoder(dummy_input)
            flattened_size = self.flat(dummy_output).shape[1]
        
        # Cabeça de classificação
        self.class_head = nn.Sequential(
            nn.Linear(flattened_size, hidden_channels[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels[0], n_targets)
        )
        
        self.name = f"CKAN_CNN_like_(gs={grid_size})"

    def _forward_features(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flat(x)
        x = self.class_head(x)
        return x

    def get_embedding(self, x):
        features = self._forward_features(x)
        features = self.flat(features)
        
        embedding = features
        for layer in self.class_head[:-1]:
            embedding = layer(embedding)
        return embedding