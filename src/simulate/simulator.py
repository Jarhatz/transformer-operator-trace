from enum import Enum
import torch
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    rotate_half
)


class TransformerOpSimulator:
    def __init__(self, operation: torch.nn.Module):
        """
        Simulates the operations of a transformer block.
        """
        self.operation = operation
        self.base_cost_per_element = 0.5

    def __call__(self, *args, **kwargs):
        """
        Call the operation and return the duration and the result.
        """
        return self.duration(*args, **kwargs), self.operation(*args, **kwargs)

    def duration(self, *args, **kwargs):
        """
        Simulates the duration (μs) of the operation using a formula based on input size and operation attributes.
        """
        x = args[0]
        numel = x.numel()
        
        if isinstance(self.operation, MatMul):
            compute_complexity = numel * args[1].shape[-2]
            matmul_efficiency = 0.001 # Assume MatMul is extremely efficient
            return int(compute_complexity * self.base_cost_per_element * matmul_efficiency)
        
        elif isinstance(self.operation, MatAdd):
            # Residual connections are simple element-wise additions
            residual_efficiency = 0.25
            return int(numel * self.base_cost_per_element * residual_efficiency)
        
        elif isinstance(self.operation, torch.nn.Linear):
            # Matrix multiplication complexity: O(N*M*K)
            compute_complexity = numel * self.operation.out_features
            matmul_efficiency = 0.001 # Assume MatMul is extremely efficient
            return int(compute_complexity * self.base_cost_per_element * matmul_efficiency)

        elif isinstance(self.operation, torch.nn.SiLU):
            # Activation functions are typically element-wise operations
            activation_cost_factor = 0.1
            return int(numel * self.base_cost_per_element * activation_cost_factor)
            
        elif isinstance(self.operation, LlamaRMSNorm):
            # Normalization involves multiple passes through the data
            # First to compute mean, then to compute variance, then to normalize
            norm_passes = 3
            norm_efficiency = 0.1
            return int(numel * self.base_cost_per_element * norm_passes * norm_efficiency)
            
        elif isinstance(self.operation, ROPE):
            # Rotary position embedding involves trigonometric functions
            # which are more expensive than basic arithmetic
            rope_complexity_factor = 0.15
            
            # RoPE is applied per head dimension
            rope_elements = numel // self.operation.num_heads
            return int(rope_elements * self.base_cost_per_element * rope_complexity_factor)
        
        elif isinstance(self.operation, SoftMax):
            # Softmax stage involves multiple passes through the data
            # First to find the max, then to exponentiate, then to normalize
            # (but we treat this as separate operation stages)
            softmax_efficiency = 0.1
            return int(numel * self.base_cost_per_element * softmax_efficiency)
        
        elif isinstance(self.operation, KVCache):
            # Updating KV cache is a simple concatenation operation
            # but assume it has a higher cost due to cache management
            kv_cache_efficiency = 0.15
            n_cache = self.operation.k_cache.numel()
            return int((numel + n_cache) * self.base_cost_per_element * kv_cache_efficiency)
        
        else:
            # Use a simple element-based formula
            default_efficiency = 0.1
            return int(numel * self.base_cost_per_element * default_efficiency)


class MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        """
        Perform matrix multiplication.
        """
        return torch.matmul(a, b)


class MatAdd(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        """
        Apply the residual connection.
        """
        return x + residual
    

class ROPE(torch.nn.Module):
    def __init__(self, cos: torch.Tensor, sin: torch.Tensor):
        super().__init__()
        self.cos, self.sin = cos.unsqueeze(1), sin.unsqueeze(1)
        self.num_heads = self.cos.shape[0]

    def forward(self, x: torch.Tensor):
        """
        Apply rotary positional embedding on input embeddings.
        """
        return (x * self.cos) + (rotate_half(x) * self.sin)


class KVCache(torch.nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.k_cache = torch.randn(1, 8, seq_len, 64)
        self.v_cache = torch.randn(1, 8, seq_len, 64)
    
    def forward(self, x: torch.Tensor, type):
        """
        Update the key-value cache.
        """
        if type == "K":
            self.k_cache = torch.cat((self.k_cache, x), dim=-2)
            return self.k_cache
        elif type == "V":
            self.v_cache = torch.cat((self.v_cache, x), dim=-2)
            return self.v_cache
        else:
            raise ValueError(f"Unknown type: {type}")

class SoftMax(torch.nn.Module):
    def __init__(self):
        super().__init__()


class SoftMax_Max(SoftMax):
    def forward(self, x: torch.Tensor):
        """
        Find the maximum value in the input tensor.
        """
        return torch.max(x, dim=-1, keepdim=True)[0]
    

class SoftMax_Exp(SoftMax):
    def forward(self, x: torch.Tensor, max):
        """
        Apply the exponential function on the input tensor.
        """
        return torch.exp(x - max)
    

class SoftMax_Norm(SoftMax):
    def forward(self, x: torch.Tensor, exp):
        """
        Normalize the input tensor.
        """
        return exp / torch.sum(exp, dim=-1, keepdim=True)
        
