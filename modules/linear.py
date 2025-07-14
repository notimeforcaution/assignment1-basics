import torch

from jaxtyping import Float, Int
from torch import Tensor
from einops import einsum


class Linear(torch.nn.Module):
    in_features: int
    out_features: int
    weight: Float[Tensor, "d_out d_in"]

    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            data=torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
