import torch
from torch import nn, Tensor, device, dtype

from jaxtyping import Float
from einops import einsum


class Linear(nn.Module):
    in_features: int
    out_features: int
    weights: Float[Tensor, "d_out d_in"]

    def __init__(
        self,
        in_features: int,  # Float[Tensor, "... d_in"]
        out_features: int,  # Float[Tensor, "... d_out"]
        device: device = None,
        dtype: dtype = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(data=torch.randn(out_features, in_features))

    def forward(self, x: Float[Tensor, "... din"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
