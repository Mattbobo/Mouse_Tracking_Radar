# -*- coding: utf-8 -*-
import os, numpy as np, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import datetime  # ← 放在檔案 import 區塊


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PLOTS = True

# ---------- NPZ Dataset ----------
class NPZDataset(Dataset):
    """
    讀取 preprocessed distance data
    npz: X:[N,T,1,H,W], y:[N]
    mean/std: 由 TRAIN 計算出的單一標量
    mode: 'train'|'val'|'test'（訓練時做些微增強）
    """
    def __init__(self, npz_path, mean, std, mode='train'):
        data = np.load(npz_path)
        self.X = data['X'].astype(np.float32)
        self.y = data['y'].astype(np.float32)
        self.mean = float(mean); self.std = float(std)
        self.mode = mode
        if self.std <= 0: raise ValueError("std 必須 > 0")

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = float(self.y[idx])

        # normalize
        x = (x - self.mean) / self.std

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

def load_mean_std(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return float(lines[0]), float(lines[1])

# ---------- Model ----------
class Net(nn.Module):
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

# ---------- Metrics ----------
def regression_metrics(y_true, y_pred, deltas=(1.,2.,5.)):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1. - ss_res/ss_tot if ss_tot>0 else 0.
    delta_acc = {f'd{int(d)}': np.mean(np.abs(y_true-y_pred)<=d) for d in deltas}
    return mae, rmse, r2, delta_acc

# ---------- Training ----------
def train_model(
    data_dir=os.path.join('Process_data','Distance_data'),
    epochs=100, batch_size=64, lr=1e-3):

    mean_tr, std_tr = load_mean_std(os.path.join(data_dir, 'mean_std.txt'))
    print(f'[TRAIN mean/std] mean={mean_tr:.6f}, std={std_tr:.6f}')

    train_ds = NPZDataset(os.path.join(data_dir, 'train.npz'), mean_tr, std_tr, mode='train')
    val_ds   = NPZDataset(os.path.join(data_dir, 'val.npz'),   mean_tr, std_tr, mode='val')
    test_ds  = NPZDataset(os.path.join(data_dir, 'test.npz'),  mean_tr, std_tr, mode='test')

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,  drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size, shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = Net().to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler()

    tr_losses, va_losses = [], []
    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds); tr_losses.append(tr_loss)

        model.eval()
        va_loss, preds, trues = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                with autocast():
                    pred = model(xb)
                    va_loss += criterion(pred, yb).item() * xb.size(0)
                preds.append(pred.cpu().numpy()); trues.append(yb.cpu().numpy())
        va_loss /= len(val_ds); va_losses.append(va_loss)

        mae, rmse, _, da = regression_metrics(np.concatenate(trues), np.concatenate(preds), deltas=(2,))
        print(f"Epoch {ep}/{epochs} | TrL {tr_loss:.4f} | VaL {va_loss:.4f} | "
              f"MAE {mae:.2f}cm | RMSE {rmse:.2f}cm | δ@2cm {da['d2']*100:.1f}%")
        scheduler.step()

    # ====== 建立儲存目錄（一次訓練一個時間戳）======
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join("model_list", "dist_model", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # 保存 loss 曲線
    if SAVE_PLOTS:
        plt.figure()
        plt.plot(tr_losses, label='Train');
        plt.plot(va_losses, label='Val')
        plt.xlabel('Epoch');
        plt.ylabel('HuberLoss');
        plt.legend();
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'dist_loss_curve.png'))
        plt.close()

    # Final test
    model.eval();
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            with autocast():
                out = model(xb)
            preds.append(out.cpu().numpy());
            trues.append(yb.numpy())
    preds = np.concatenate(preds);
    trues = np.concatenate(trues)
    mae, rmse, r2, da = regression_metrics(trues, preds, deltas=(1., 2., 5.))
    print(f"TEST => MAE {mae:.2f}cm, RMSE {rmse:.2f}cm, R2 {r2:.3f}, "
          f"δ@1cm {da['d1'] * 100:.1f}%, δ@2cm {da['d2'] * 100:.1f}%, δ@5cm {da['d5'] * 100:.1f}%")

    # 保存 metrics 與模型
    with open(os.path.join(save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"MAE,{mae:.4f}\nRMSE,{rmse:.4f}\nR2,{r2:.4f}\n")
        for k, v in da.items():
            f.write(f"{k},{v:.4f}\n")

    torch.save(model.state_dict(), os.path.join(save_dir, 'dist_model_seq.pth'))
    print(f"距離模型、loss 曲線與評估已保存到：{save_dir}")

if __name__ == '__main__':
    train_model()
