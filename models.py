import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class VisionTransformer(nn.Module):
    def __init__(self, num_classes, hidden_size=768, num_layers=12, num_heads=12, dropout=0.1):
        """
        Initialize the Vision Transformer model.

        Args:
        - num_classes (int): Number of output classes.
        - hidden_size (int): Size of the transformer's hidden layers (default: 768).
        - num_layers (int): Number of transformer layers (default: 12).
        - num_heads (int): Number of attention heads in each layer (default: 12).
        - dropout (float): Dropout probability (default: 0.1).

        """
        super(VisionTransformer, self).__init__()
        
        # Define ViT configuration
        config = ViTConfig(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        # Initialize ViT model
        self.vit = ViTModel(config)

        # Linear classifier for classification
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, x):
        """
        Forward pass of the Vision Transformer model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
        - logits (torch.Tensor): Output logits of shape (batch_size, num_classes).

        """
        # Perform forward pass through ViT
        outputs = self.vit(pixel_values=x)
        
        # Get pooled output from ViT
        logits = self.classifier(outputs.pooler_output)
        
        return logits

    def freeze_encoder(self):
        """
        Freeze the parameters of the Vision Transformer encoder.

        """
        for param in self.vit.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """
        Unfreeze the parameters of the Vision Transformer encoder.

        """
        for param in self.vit.parameters():
            param.requires_grad = True
