import math
import torch
from torch import nn
from einops import rearrange, einsum, reduce
from jaxtyping import Float, Int

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight: Float[torch.Tensor, "d_out, d_in"]= nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        self._init_weights(mean=0, std=math.sqrt(2 / (in_features + out_features)))

    def _init_weights(self, mean: float, std: float):
        nn.init.trunc_normal_(
            self.weight, 
            mean=mean, 
            std=std, 
            a=-3 * std, 
            b=3 * std
        )

    def forward(self, x: Float[torch.Tensor, "... d_in"]) -> Float[torch.Tensor, "... d_out"]:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.weight: Float[torch.Tensor, "vocab_size, d_model"] = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        self._init_weights(mean=0, std=1)
    
    def _init_weights(self, mean: float, std: float):
        nn.init.trunc_normal_(
            self.weight, 
            mean=mean, 
            std=std, 
            a=-std, 
            b=std
        )

    def forward(self, token_ids: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_model"]:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight: Float[torch.Tensor, "d_model"] = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.ones_(self.weight)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32) # prevent overflow
        result = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps) * x * self.weight
        #result = torch.rsqrt(reduce(x.pow(2), "... d_model -> ... 1", "mean") + self.eps) * x * self.weight
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff, device, dtype)
        self.linear2 = Linear(d_ff, d_model, device, dtype)
        self.linear3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Float[torch.Tensor, "... d_model"]) -> Float[torch.Tensor, "... d_model"]:
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))

def silu(x: Float[torch.Tensor, "..."]) -> Float[torch.Tensor, "..."]:
    return x * torch.sigmoid(x)




if __name__ == "__main__":
    pass
    import torch.nn.functional as F
    F.silu()