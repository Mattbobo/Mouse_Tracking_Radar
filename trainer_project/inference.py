import pickle
import zmq
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import torch.nn.functional as F
import numpy as np
from gesture_file.model import CNN3DBackbone

# ---------- Model Definitions ----------
class AngleNet(nn.Module):
    def __init__(self, in_ch=2, conv_channels=[32,64,128,256],
                 hidden_size=64, num_layers=1):
        super().__init__()
        # 普通卷积块：Conv → BN → ReLU
        class ConvBlock(nn.Module):
            def __init__(self, in_c, out_c):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                )
            def forward(self, x):
                return self.conv(x)

        # 构造多层卷积
        self.blocks = nn.ModuleList()
        prev = in_ch
        for ch in conv_channels:
            self.blocks.append(ConvBlock(prev, ch))
            prev = ch

        # 全局池化 + RNN + 回归头保持不变
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.GRU(input_size=prev,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden_size//2, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        # (B*T, C, H, W)
        x = x.view(B*T, C, H, W)

        for block in self.blocks:
            x = block(x)
            h, w = x.shape[-2:]
            # 空间下采样
            if h >= 2 and w >= 2:
                x = F.max_pool2d(x, 2)

        # 全局池化到 (B*T, C, 1, 1) → (B, T, C)
        x = self.global_pool(x).view(B, T, -1)
        # GRU + 取最后时刻输出
        out, _ = self.rnn(x)
        # 回归输出
        return self.head(out[:, -1]).squeeze(1)


class DistNet(nn.Module):
    """只輸出距離的模型，與 trainer 中 Net 一致"""
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
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

# ---------- Inference Server ----------

def infer_gesture(seq, gesture_model, prototypes, proto_labels, thr_gesture, device):
    """
    seq: (T, 2, 32, 32)
    return: raw_label (0/1)
    """
    x = torch.from_numpy(seq.astype('float32')).permute(1, 0, 2, 3).unsqueeze(0).to(device)
    # (T,2,H,W) -> (1,2,T,H,W)

    with torch.no_grad(), autocast():
        z = gesture_model(x)                 # (1, D)
        dist = torch.cdist(z, prototypes)    # (1, K)
        d_min, k = torch.min(dist, dim=1)
        d_min_val = float(d_min.item())
        c_hat = int(proto_labels[int(k.item())])

        raw_label = 0
        if c_hat == 1 and d_min_val < thr_gesture:
            raw_label = 1

    return raw_label

def main():
    # ZeroMQ REP socket
    ctx  = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind("tcp://*:5555")
    print("[Inference Server] Listening on tcp://*:5555")

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate models
    angle_model = AngleNet().to(device)
    dist_model  = DistNet().to(device)

    # Load trained weights
    angle_model.load_state_dict(torch.load(r'C:\Users\mc2\Desktop\degree_trainer\model_list\angle_model\2025-09-02_14-06-50\angle_model_seq.pth', map_location=device))
    dist_model .load_state_dict(torch.load(r'C:\Users\mc2\Desktop\degree_trainer\model_list\dist_model\2025-09-02_14-10-40\dist_model_seq.pth',  map_location=device))

    #gesture load
    gesture_model = CNN3DBackbone(in_channels=2, emb_dim=32).to(device)
    gesture_state = torch.load(r'gesture_file\model_last.pth', map_location=device)
    gesture_model.load_state_dict(gesture_state, strict=True)
    gesture_model.eval()

    proto_npz = np.load(r'gesture_file\prototypes_last.npz')
    prototypes = torch.tensor(proto_npz["prototypes"], dtype=torch.float32).to(device)

    K = prototypes.shape[0]
    proto_labels = [1][:K]
    if "gesture_classes" in proto_npz:
        gc = list(np.array(proto_npz["gesture_classes"]).tolist())
        if len(gc) == K:
            proto_labels = gc
        elif len(gc) == K - 1:
            proto_labels = [0] + gc

    thr_gesture = 0.7

    angle_model.eval()
    dist_model.eval()

    while True:
        try:
            mode, seq = pickle.loads(sock.recv())

            angle, dist, gesture = None, None, None

            # Angle inference
            if mode in ('both', 'angle'):
                x = torch.from_numpy(seq.astype('float32')).unsqueeze(0).to(device)
                with torch.no_grad(), autocast():
                    angle = angle_model(x).item()

            # Distance inference
            if mode in ('both', 'dist'):
                x = torch.from_numpy(seq.astype('float32')).unsqueeze(0).to(device)
                if mode == 'both':
                    x_dist = x[:, :, 0:1, :, :].contiguous()
                else:
                    x_dist = x
                with torch.no_grad(), autocast():
                    dist = dist_model(x_dist).item()

            # Gesture inference
            if mode == 'gesture':
                gesture = infer_gesture(
                    seq=seq,
                    gesture_model=gesture_model,
                    prototypes=prototypes,
                    proto_labels=proto_labels,
                    thr_gesture=thr_gesture,
                    device=device
                )

            # reply
            if mode == 'gesture':
                sock.send_pyobj(gesture)
            else:
                sock.send_pyobj((angle, dist))

        except Exception as e:
            print(f"[Inference Error] {e}")
            if mode == 'gesture':
                sock.send_pyobj(0)
            else:
                sock.send_pyobj((None, None))

if __name__ == '__main__':
    main()
