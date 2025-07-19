import torch
from torch import embedding, nn, Tensor, device, dtype

from jaxtyping import Float, Int


class Embedding(nn.Module):
    # Contains an embedding of len embedding_dim for each index.
    embedding_matrix: Float[Tensor, "num_embeddings embedding_dim"]

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ):
        super().__init__()
        self.embedding_matrix = nn.Parameter(
            data=torch.zeros(num_embeddings, embedding_dim)
        )

    def forward(
        self, token_ids: Int[Tensor, "seq_len"]
    ) -> Float[Tensor, "seq_len d_model"]:
        # batch_size, seq_len --> batch_size, seq_len, d_model
        return self.embedding_matrix[token_ids]
