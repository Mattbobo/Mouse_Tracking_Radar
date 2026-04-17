# models.py  (for training.py use)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------
# 3D-CNN Encoder: (3 conv blocks + GN + ReLU + MaxPool) → GMP → FC(embedding)
# ---------------------------------------------------------------


class CNN3DBackbone(nn.Module):
    """
    三層 3D CNN Backbone：
      Input:  (N, C=2, T=40, H=32, W=32)
      Output: (N, emb_dim)
    """
    def __init__(self, in_channels: int = 2, base: int = 16,
                 emb_dim: int = 32, gn_groups: int = 8):
        super().__init__()
        C1, C2, C3 = base, base * 2, base * 4

        # Block 1
        self.conv1 = nn.Conv3d(in_channels, C1, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=min(gn_groups, C1), num_channels=C1)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv3d(C1, C2, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=min(gn_groups, C2), num_channels=C2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Block 3
        self.conv3 = nn.Conv3d(C2, C3, kernel_size=3, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(num_groups=min(gn_groups, C3), num_channels=C3)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # Global Max Pool + FC embedding
        self.gmp = nn.AdaptiveMaxPool3d((3, 1, 1))
        self.fc = nn.Linear(C3 * 3, emb_dim)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = self.pool2(F.relu(self.gn2(self.conv2(x))))
        x = self.pool3(F.relu(self.gn3(self.conv3(x))))
        x = self.gmp(x).flatten(1)
        x = self.fc(x)
        return x


# -----------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------
if __name__ == "__main__":
    model = CNN3DBackbone(in_channels=2, emb_dim=32)
    x = torch.randn(4, 2, 20, 32, 32)  # (N, C, T, H, W)
    z = model(x)
    print("Embedding shape:", z.shape)
