import torch
import torch.nn.functional as F
import torch.nn as nn
from torchinfo import summary


# ---------- 工具 1：逐步列印形狀 ----------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.rnn  = nn.GRU(input_size=128, hidden_size=32, batch_first=True)
        self.head = nn.Linear(32, 1)

    def forward(self, x):      # x: (B,T,1,H,W)
        B,T,C,H,W = x.shape
        x = x.view(B*T, C, H, W)
        f = self.backbone(x).view(B, T, -1)
        out,_ = self.rnn(f)
        return self.head(out[:, -1]).squeeze(1)

@torch.no_grad()
def print_architecture(model, input_shape=(2, 20, 2, 32, 32)):
    """
    以假資料跑一次 forward，逐層列出張量形狀變化。
    input_shape: (B, T, C, H, W)
    """
    model.eval()
    B, T, C, H, W = input_shape
    x = torch.zeros(input_shape, dtype=torch.float32)

    lines = []
    lines.append(f"Input  (B,T,C,H,W): {tuple(x.shape)}")
    x = x.view(B*T, C, H, W)
    lines.append(f"Reshape → (B*T,C,H,W): {tuple(x.shape)}")

    # CNN blocks + 可用的 2x2 MaxPool
    for i, block in enumerate(model.blocks, 1):
        x = block(x)
        lines.append(f"[Block {i}] Conv3x3→BN→ReLU  → {tuple(x.shape)}")
        h, w = x.shape[-2:]
        if h >= 2 and w >= 2:
            x = F.max_pool2d(x, 2)
            lines.append(f"          MaxPool2d(2)     → {tuple(x.shape)}")

    # Global Pool → 還原時間
    x = model.global_pool(x)          # (B*T, C_out, 1, 1)
    lines.append(f"AdaptiveAvgPool2d(1) → {tuple(x.shape)}")
    x = x.view(B, T, -1)
    lines.append(f"Reshape → (B,T,C_out): {tuple(x.shape)}")

    # GRU
    out, _ = model.rnn(x)
    lines.append(f"GRU(input={x.shape[-1]}, hidden={model.rnn.hidden_size}, "
                 f"layers={model.rnn.num_layers}) → {tuple(out.shape)}")
    lines.append(f"Select last timestep → (B,{model.rnn.hidden_size})")

    # Head
    hidden = model.rnn.hidden_size
    hidden2 = model.head[0].out_features  # 第一個 Linear 的輸出維度
    lines.append(f"Head: Linear({hidden},{hidden2}) → ReLU → Dropout(0.15) "
                 f"→ Linear({hidden2},1) → (B,)")

    print("\n".join(lines))


# ---------- 工具 2：像圖一樣的文字方塊 ----------
def print_text_diagram(model, frame_hw=(32, 32), T=20, C_in=2):
    """
    產生簡潔的「方塊式」架構文字圖，和你附圖一致的風格。
    frame_hw: 每幀 H,W
    T: 序列長度
    C_in: 每幀輸入通道數
    """
    H, W = frame_hw
    ch_list = [blk.conv[0].out_channels for blk in model.blocks]
    pool_marks = []
    # 模擬池化次數（遇到 h,w>=2 就會池化、尺寸對半）
    h, w = H, W
    for _ in ch_list:
        if h >= 2 and w >= 2:
            h //= 2
            w //= 2
            pool_marks.append(True)
        else:
            pool_marks.append(False)

    lines = []
    def box(s): return s  # 可以改成加邊框符號，這裡保持簡潔

    lines.append(box(f"Input  (B,T,C,H,W) = (B,{T},{C_in},{H},{W})"))
    for i, ch in enumerate(ch_list, 1):
        lines.append(box(f"3x3 Conv2d, {ch}"))
        lines.append(box("BatchNorm2d"))
        lines.append(box("ReLU"))
        if pool_marks[i-1]:
            lines.append(box("2x2 MaxPooling2d"))

    lines.append(box("1x1 AdaptiveAvgPool2d"))
    lines.append(box(f"GRU, hidden={model.rnn.hidden_size}, layers={model.rnn.num_layers}"))
    head_hidden = model.head[0].out_features
    lines.append(box(f"fc {head_hidden}"))
    lines.append(box("ReLU"))
    lines.append(box("Dropout, 0.15"))
    lines.append(box("fc 1"))

    print("\n".join(lines))


# ------------------ 範例用法 ------------------
if __name__ == "__main__":
    # # 這裡假設你的 Net 定義已經在作用域中
    # net = Net()
    #
    # # 1) 列出逐步形狀
    # print_architecture(net, input_shape=(4, 20, 1, 32, 32))  # B=4, T=20
    #
    # print("\n" + "="*60 + "\n")
    #
    # # 2) 列出簡潔文字方塊圖
    # print_text_diagram(net, frame_hw=(32, 32), T=20, C_in=1)

    model = Net()
    summary(model, input_size=(4, 20, 1, 32, 32))  # (B,T,C,H,W)