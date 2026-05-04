import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self, in_channels: int = 3, patch_size: int = 4, emb_size: int = 128
    ) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        emb_size: int = 128,
        num_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layernorm1 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(
            emb_size, num_heads, dropout=dropout, batch_first=True
        )

        self.layernorm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layernorm1(x)
        x, _ = self.attention(x, x, x)
        x = x + residual

        residual = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = x + residual
        return x


class ViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        emb_size: int = 128,
        img_size: int = 32,
        depth: int = 6,
        num_heads: int = 8,
        n_classes: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)

        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    emb_size=emb_size,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

        self.layernorm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.positions

        for layer in self.layers:
            x = layer(x)

        x = self.layernorm(x[:, 0])
        return self.head(x)
