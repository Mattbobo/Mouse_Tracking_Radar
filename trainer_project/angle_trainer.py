import os, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
import datetime

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PLOTS = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled   = True

# ---------- Dataset for preprocessed NPZ ----------
class NPZDataset(Dataset):
    """
    讀取預先切好視窗的資料 (X:[N,T,C,H,W], y:[N])
    並依據給定 mean/std 做正規化。
    mode: 'train'/'val'/'test' 決定是否套用增強。
    """
    def __init__(self, npz_path, mean, std, mode='train'):
        super().__init__()
        data = np.load(npz_path)
        self.X = data['X'].astype(np.float32)  # (N,T,C,H,W)
        self.y = data['y'].astype(np.float32)  # (N,)
        self.mean = float(mean)
        self.std  = float(std)
        self.mode = mode
        if self.std <= 0:
            raise ValueError("std 必須 > 0")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        # normalization
        x = (x - self.mean) / self.std

        return torch.from_numpy(x), torch.tensor(float(y), dtype=torch.float32)

# ---------- Utility to read mean/std ----------
def load_mean_std(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    mean = float(lines[0]); std = float(lines[1])
    return mean, std

# ---------- Model Definition (unchanged) ----------
class Net(nn.Module):
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


# ---------- Metrics (unchanged) ----------
def regression_metrics(y_true, y_pred, deltas=(1.,2.,5.)):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot>0 else 0.
    da = {f'd{int(d)}': np.mean(np.abs(y_true-y_pred)<=d) for d in deltas}
    return mae, rmse, r2, da

# ---------- Training Pipeline w/ AMP & CosineAnnealingLR ----------
def train_model(
    data_dir=os.path.join('Process_data','Angle_data'),
    epochs=100, batch_size=64, lr=1e-3):

    # === 讀取 mean/std ===
    mean_all, std_all = load_mean_std(os.path.join(data_dir, 'mean_std.txt'))
    print(f'Precomputed mean/std => mean:{mean_all:.6f}, std:{std_all:.6f}')

    # === 建立 Dataset / DataLoader ===
    train_ds = NPZDataset(os.path.join(data_dir, 'train.npz'), mean_all, std_all, mode='train')
    val_ds   = NPZDataset(os.path.join(data_dir, 'val.npz'),   mean_all, std_all, mode='val')
    test_ds  = NPZDataset(os.path.join(data_dir, 'test.npz'),  mean_all, std_all, mode='test')

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)

    # 5) model/opt/scaler/scheduler
    model = Net().to(DEVICE)
    print(f'Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    criterion = nn.HuberLoss(delta=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = GradScaler()

    # 6) train & val
    train_losses, val_losses = [], []
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb,yb in train_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds); train_losses.append(tr_loss)

        model.eval()
        val_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for xb,yb in val_loader:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                with autocast():
                    pred = model(xb)
                    v = criterion(pred, yb).item() * xb.size(0)
                val_loss += v
                preds.append(pred.cpu().numpy()); trues.append(yb.cpu().numpy())
        val_loss /= len(val_ds); val_losses.append(val_loss)

        mae, rmse, _, da = regression_metrics(
            np.concatenate(trues), np.concatenate(preds), deltas=(2,))
        print(f'Epoch {ep}/{epochs} | TrL {tr_loss:.4f} | VaL {val_loss:.4f} | '
              f'MAE {mae:.2f} | RMSE {rmse:.2f} | δ@2° {da["d2"]*100:.1f}%')
        scheduler.step()

    # 7) 保存 loss 曲線
    if SAVE_PLOTS:
        plt.figure()
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Val')
        plt.xlabel('Epoch');
        plt.ylabel('HuberLoss')
        plt.legend();
        plt.tight_layout()

        # 建立儲存資料夾路徑
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join("model_list", "angle_model", timestamp)
        os.makedirs(save_dir, exist_ok=True)

        loss_curve_path = os.path.join(save_dir, 'angle_loss_curve.png')
        plt.savefig(loss_curve_path)
        plt.close()

    # 8) 最終測試
    model.eval();
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            with autocast():
                out = model(xb)
            preds.append(out.cpu().numpy());
            trues.append(yb.numpy())
    preds, trues = np.concatenate(preds), np.concatenate(trues)
    mae, rmse, r2, da = regression_metrics(trues, preds, deltas=(1., 2., 5.))
    print(f'\nTEST => MAE {mae:.2f}°, RMSE {rmse:.2f}°, R² {r2:.3f}, '
          f'δ@1° {da["d1"] * 100:.1f}%, δ@2° {da["d2"] * 100:.1f}%, δ@5° {da["d5"] * 100:.1f}%')

    # 9) 保存指標與模型
    metrics_path = os.path.join(save_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'MAE,{mae:.4f}\nRMSE,{rmse:.4f}\nR2,{r2:.4f}\n')
        for k, v in da.items():
            f.write(f'{k},{v:.4f}\n')

    model_path = os.path.join(save_dir, 'angle_model_seq.pth')
    torch.save(model.state_dict(), model_path)

    print(f'模型、loss 曲線與評估指標已保存到 {save_dir}')

if __name__ == '__main__':
    train_model()
