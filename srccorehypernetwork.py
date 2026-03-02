"""
Hypernetwork Module - Dynamically generates task-specific neural networks
Architecture Choice: We use a meta-network that generates weights for task-specific networks,
enabling rapid adaptation to different market regimes without retraining the entire model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class HyperNetwork(nn.Module):
    """Meta-network that generates weights for task-specific networks"""
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dims: List[int] = [256, 512, 256],
        task_input_dim: int = 64,
        task_output_dim: int = 3,
        task_hidden_dims: List[int] = [128, 64],
        dropout_rate: float = 0.2
    ):
        """
        Initialize HyperNetwork
        
        Args:
            latent_dim: Dimension of latent market regime vector
            hidden_dims: Dimensions of hypernetwork hidden layers
            task_input_dim: Input dimension for generated task network
            task_output_dim: Output dimension for generated task network
            task_hidden_dims: Hidden layer dimensions for task network
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.task_hidden_dims = task_hidden_dims
        self.task_input_dim = task_input_dim
        self.task_output_dim = task_output_dim
        
        # Build hypernetwork layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.hyper_layers = nn.Sequential(*layers)
        
        # Calculate total parameters needed for task network
        self.total_task_params = self._calculate_task_params()
        
        # Final layer to generate all task network weights
        self.weight_generator = nn.Linear(
            prev_dim,
            self.total_task_params
        )
        
        logger.info(f"HyperNetwork initialized with latent_dim={latent_dim}, "
                   f"generating {self.total_task_params} task parameters")
    
    def _calculate_task_params(self) -> int:
        """Calculate total parameters needed for task network"""
        total_params = 0
        
        # Input layer
        total_params += (self.task_input_dim * self.task_hidden_dims[0] 
                        + self.task_hidden_dims[0])
        
        # Hidden layers
        for i in range(len(self.task_hidden_dims) - 1):
            total_params += (self.task_hidden_dims[i] * self.task_hidden_dims[i + 1]
                           + self.task_hidden_dims[i + 1])
        
        # Output layer
        total_params += (self.task_hidden_dims[-1] * self.task_output_dim
                        + self.task_output_dim)
        
        return total_params
    
    def forward(self, latent_vector: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate weights for task network
        
        Args:
            latent_vector: Market regime embedding [batch_size, latent_dim]
            
        Returns:
            Dictionary containing generated weights for task network
        """
        if latent_vector.dim() != 2:
            raise ValueError(f"latent_vector must be 2D, got shape {latent_vector.shape}")
        
        batch_size = latent_vector.size(0)
        
        # Generate weight vector
        features = self.hyper_layers(latent_vector)
        weight_vector = self.weight_generator(features)
        
        # Split weight vector into individual weight matrices and biases
        weights_dict = self._split_weights(weight_vector, batch_size)
        
        return weights_dict
    
    def _split_weights(
        self, 
        weight_vector: torch.Tensor, 
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Split flat weight vector into structured weight matrices"""
        weights_dict = {}
        idx = 0
        
        # Input layer weights
        input_weights_size = self.task_input_dim * self.task_hidden_dims[0]
        input_bias_size = self.task_hidden_dims[0]
        
        weights_dict['layer1.weight'] = weight_vector[:, idx:idx+input_weights_size].reshape(
            batch_size, self.task_hidden_dims[0], self.task_input_dim
        )
        idx += input_weights_size
        
        weights_dict['layer1.bias'] = weight_vector[:, idx:idx+input_bias_size]
        idx += input_bias_size
        
        # Hidden layer weights
        for i in range(len(self.task