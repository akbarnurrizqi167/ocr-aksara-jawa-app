"""
Model Architecture — CRNN Hybrid untuk OCR Aksara Jawa

Arsitektur:
    1. VGG16 Feature Extractor (Block 1-3, frozen)
    2. Adaptation Layers (Conv2d + BN + Pool)
    3. Bidirectional LSTM (2 layers)
    4. Fully Connected Output Layer

Input:  (batch, 3, 32, 128)
Output: (batch, 16, num_classes)
"""

import torch
import torch.nn as nn
from torchvision import models


class VGG16FeatureExtractor(nn.Module):
    """
    VGG16 Block 1-3 sebagai Feature Extractor
    Input:  (batch, 3, 32, 128)
    Output: (batch, 256, 4, 16)
    """

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()

        if pretrained:
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            vgg16 = models.vgg16(weights=None)

        # Block 1-3 (index 0-16)
        self.features = nn.Sequential(*list(vgg16.features.children())[:17])

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class AdaptationLayers(nn.Module):
    """
    Adaptation Layers untuk domain aksara Jawa
    Input:  (batch, 256, 4, 16)
    Output: (batch, 16, 512)
    """

    def __init__(self, in_channels: int = 256, hidden_channels: int = 512, dropout: float = 0.2):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

        self.height_pool = nn.AdaptiveMaxPool2d((1, None))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.height_pool(x)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)
        return x


class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM untuk sequence modeling
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        output = self.dropout(output)
        return output


class CRNNHybrid(nn.Module):
    """
    CRNN Hybrid Model untuk OCR Aksara Jawa

    Arsitektur:
        1. VGG16 Block 1-3 (Frozen)
        2. Adaptation Layers (Trainable)
        3. BiLSTM (Trainable)
        4. FC Layer (Trainable)

    Input:  (batch, 3, 32, 128)
    Output: (batch, 16, num_classes)
    """

    def __init__(
        self,
        num_classes: int,
        pretrained_vgg: bool = True,
        freeze_vgg: bool = True,
        hidden_size: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        self.num_classes = num_classes

        self.feature_extractor = VGG16FeatureExtractor(
            pretrained=pretrained_vgg,
            freeze=freeze_vgg
        )

        self.adaptation = AdaptationLayers(
            in_channels=256,
            hidden_channels=hidden_size * 2,
            dropout=dropout
        )

        self.bilstm = BidirectionalLSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        adapted = self.adaptation(features)
        sequence = self.bilstm(adapted)
        output = self.fc(sequence)
        return output

    def get_sequence_length(self) -> int:
        return 16
