import functools
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultitaskAutoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        super(MultitaskAutoencoder, self).__init__()
        
        layers = []
        in_size = input_size
        self.hidden_layers = torch.nn.ModuleList()  # Store hidden layers to access embeddings
        
        # Add hidden layers (Encoder)
        for hidden_size in hidden_layer_sizes:
            self.hidden_layers.append(torch.nn.Linear(in_size, hidden_size))
            self.hidden_layers.append(torch.nn.ReLU())
            in_size = hidden_size
        
        # Add output layer for classification
        self.output_layer = torch.nn.Linear(in_size, output_size)
        
        # Decoder part (to reconstruct the input)
        self.decoder_layers = torch.nn.ModuleList()
        for hidden_size in reversed(hidden_layer_sizes):
            self.decoder_layers.append(torch.nn.Linear(in_size, hidden_size))
            self.decoder_layers.append(torch.nn.ReLU())
            in_size = hidden_size
        
        # Final decoder layer to reconstruct the original input
        self.reconstruction_layer = torch.nn.Linear(in_size, input_size)

    def forward(self, x, embeddings=False):
        # Encoder: Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        if embeddings:
            return x  # Return classification output and embeddings

        # Classification output
        class_output = self.output_layer(x)
            
        # Decoder: Reconstruct the input
        reconstructed = x
        for layer in self.decoder_layers:
            reconstructed = layer(reconstructed)
        
        reconstructed = self.reconstruction_layer(reconstructed)
        
        return class_output, reconstructed
  

class ConvAutoencoderMultitask(nn.Module):
    def __init__(self, input_height, input_width, num_classes, latent_dim_size=128, dropout_rate=0.5):
        super(ConvAutoencoderMultitask, self).__init__()
    
        self.input_height = input_height
        self.input_width = input_width
        self.dropout_rate = dropout_rate

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # Layer 1
            nn.ReLU(),
            # nn.MaxPool2d((1, 2)),  # Lateral pooling
            nn.Dropout2d(self.dropout_rate),  # Dropout for convolutional layers (spatial dropout)

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # Layer 2
            nn.ReLU(),
            # nn.MaxPool2d((1, 2)),  # Lateral pooling
            nn.Dropout2d(self.dropout_rate),  # Dropout

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),  # Layer 3
            nn.ReLU(),
            # nn.MaxPool2d((1, 2)),  # Lateral poolings
            nn.Dropout2d(self.dropout_rate),  # Dropout

            nn.Conv2d(64, latent_dim_size, kernel_size=5, stride=2, padding=2),  # Layer 4
            nn.ReLU(),
            # nn.MaxPool2d((1, 2)),  # Lateral pooling
            nn.Dropout2d(self.dropout_rate),  # Dropout
        )

        # Dynamically calculate the output dimensions
        dummy_input = torch.zeros(1, 1, input_height, input_width)
        dummy_output = self.encoder(dummy_input)
        final_height, final_width = dummy_output.size(2), dummy_output.size(3)
        self.final_height = final_height
        self.final_width = final_width
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim_size * final_height * final_width, 100),  # Layer 5
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),  # Dropout for dense layers

            nn.Linear(100, num_classes)   # Layer 6
        )
        
        # # Decoder
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=5, stride=(1, 2), padding=2, output_padding=(0, 1)),  # Adjust for lateral pooling
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout_rate),
        #     nn.ConvTranspose2d(64, 32, kernel_size=5, stride=(1, 2), padding=2, output_padding=(0, 1)),  # Adjust for lateral pooling
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout_rate),
        #     nn.ConvTranspose2d(32, 16, kernel_size=5, stride=(1, 2), padding=2, output_padding=(0, 1)),  # Adjust for lateral pooling
        #     nn.ReLU(),
        #     nn.Dropout2d(dropout_rate),
        #     nn.ConvTranspose2d(16, 1, kernel_size=5, stride=(1, 2), padding=2, output_padding=(0, 1)),  # Adjust for lateral pooling
        #     nn.Sigmoid()
        # )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim_size, 64, kernel_size=5, stride=2, padding=2, output_padding=1),  # Layer 4 inverse
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),  # Dropout

            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),   # Layer 3 inverse
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),  # Dropout

            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),   # Layer 2 inverse
            nn.ReLU(),
            nn.Dropout2d(self.dropout_rate),  # Dropout

            nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),    # Layer 1 inverse,
            nn.Sigmoid()  # Use sigmoid to scale the output to [0,1]
        )


    def calculate_final_dimensions(self):
        height, width = self.input_height, self.input_width
        for _ in range(4):  # Four convolutional layers
            height = math.ceil((height + 4 - 5) / 2 + 1)
            width = math.ceil((width + 4 - 5) / 2 + 1)
        return height, width

    def forward(self, x, embeddings=False):
        x = self.encoder(x)
        x_flat = x.view(x.size(0), -1)  # Flatten the output for the classifier
        if embeddings:
            return x_flat
        reconstructed = self.decoder(x)

        class_output = self.classifier(x_flat)
        

        return class_output, reconstructed



class MultitaskUNet(nn.Module):
    def __init__(self, input_height, input_width, num_classes, latent_dim_size=128, dropout_rate=0.5):
        in_channels = 1
        out_channels = 1
        super(MultitaskUNet, self).__init__()
        self.height = input_height
        self.dropout_rate = dropout_rate
        # Encoder
        self.encoder1 = self.conv_block(in_channels, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, latent_dim_size)
        # Bottleneck
        bottle_up = min(input_height//16, 2)
        self.bottleneck = self.conv_block(latent_dim_size, latent_dim_size)
        self.up_bottleneck = nn.ConvTranspose2d(
            in_channels=latent_dim_size, 
            out_channels=latent_dim_size, 
            kernel_size=(bottle_up, 2),  # To expand the width from 1 to 2
            stride=(bottle_up, 2),       # Keep the height dimension the same (1)
            padding=(0, 0)       # No padding needed, as we want to directly go from 1 to 2
        )
        
        # Decoder
        self.decoder4 = self.upconv_block(latent_dim_size + latent_dim_size, 64)  # concat + decoder channel size
        self.decoder3 = self.upconv_block(64 + 64, 32)   # concat + decoder channel size
        self.decoder2 = self.upconv_block(32 + 32, 16)   # concat + decoder channel size
        self.decoder1 = self.upconv_block(16 + 16, 16)        # Final output channels
        
        # Final layer
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

        # Classification head
        self.classification_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latent_dim_size * 1 * 1, 32),  # Flatten the bottleneck output, adjust size if necessary
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes)   # Output number of classes
        )
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate),  # Dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=(1,2), padding=(2,2)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate)  # Dropout
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=(1, 1)),
            nn.Dropout2d(self.dropout_rate),  # Dropout
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=5, stride=(1,2), padding=(2,2), output_padding=(0,1)),
            nn.Dropout2d(self.dropout_rate)  # Dropout
        )
    
    def forward(self, x, embeddings=False):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Bottleneck
        bottleneck = self.bottleneck(e4)
        bottleneck_up = self.up_bottleneck(bottleneck)

        # Decoder with skip connections (concatenation)
        d4 = self.decoder4(torch.cat((bottleneck_up, e4), dim=1))  # Concatenate skip connection
        d3 = self.decoder3(torch.cat((d4, e3), dim=1))
        d2 = self.decoder2(torch.cat((d3, e2), dim=1))
        d1 = self.decoder1(torch.cat((d2, e1), dim=1))
        
        # Final output
        out = self.final_conv(d1)
        if embeddings:
            return bottleneck.view(x.size(0), -1)

        # Classification output
        # print(bottleneck.shape)

        classification_output = self.classification_fc(bottleneck)
        
        return classification_output, out
    