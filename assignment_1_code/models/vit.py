import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """이미지를 패치로 나누고 벡터로 투영합니다."""
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        # Conv2d를 사용하여 패치 분할과 선형 투영을 동시에 수행 (trick)
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, 32, 32) -> (B, emb_size, 8, 8)
        x = self.projection(x)
        # (B, emb_size, 8, 8) -> (B, emb_size, 64) -> (B, 64, emb_size)
        x = x.flatten(2).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    """Transformer의 인코더 블록입니다."""
    def __init__(self, emb_size=128, num_heads=4, expansion=4, dropout=0.1):
        super().__init__()
        self.layernorm1 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)
        
        self.layernorm2 = nn.LayerNorm(emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
        )

    def forward(self, x):
        # Attention + Residual Connection
        res = x
        x = self.layernorm1(x)
        x, _ = self.attention(x, x, x)
        x = x + res
        
        # MLP + Residual Connection
        res = x
        x = self.layernorm2(x)
        x = self.mlp(x)
        x = x + res
        return x

class ViT(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, emb_size=128, img_size=32, depth=6, n_classes=10, **kwargs):
        super().__init__()
        # 1. Patch Embedding
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        
        # 2. Class Token (학습 가능한 파라미터)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        
        # 3. Positional Embedding
        num_patches = (img_size // patch_size) ** 2
        self.positions = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        
        # 4. Transformer Layers
        self.layers = nn.ModuleList([
            TransformerBlock(emb_size, **kwargs) for _ in range(depth)
        ])
        
        # 5. Classification Head
        self.ln = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        b = x.shape[0]
        x = self.patch_embedding(x)
        
        # 클래스 토큰 확장 및 결합
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) # (B, 65, emb_size)
        
        # 위치 정보 추가
        x = x + self.positions
        
        # Transformer 인코더 통과
        for layer in self.layers:
            x = layer(x)
            
        # 첫 번째 토큰(Class Token)만 사용하여 분류
        x = self.ln(x[:, 0])
        return self.head(x)