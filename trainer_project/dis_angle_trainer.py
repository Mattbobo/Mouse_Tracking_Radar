import os
import glob
import random
import h5py
import numpy as np
import torch
import matplotlib

# 1. cuDNN 自动调优 & reproducibility
torch.backends.cudnn.benchmark = True
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PLOTS = True


# ---------- Metrics ----------
def regression_metrics(y_true, y_pred, deltas=(1., 2., 5.)):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1. - ss_res / ss_tot if ss_tot > 0 else 0.
    delta_acc = {f'd{int(d)}': np.mean(np.abs(y_true - y_pred) <= d) for d in deltas}
    return mae, rmse, r2, delta_acc


# ---------- Sequence Dataset ----------
class WindowDataset(Dataset):
    """
    windows: list of (x_seq: np.ndarray (T,C,H,W), (angle, dist))
    mean, std: np.ndarray shape (C,1,1)
    """

    def __init__(self, windows, mean, std):
        self.windows = windows
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x_seq, (y_angle, y_dist) = self.windows[idx]
        x = (x_seq.astype(np.float32) - self.mean) / self.std
        y = np.array([y_angle, y_dist], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# ---------- Data Loading ----------
def load_all_data(root_dir='Record'):
    xs_list, angles_list, dists_list = [], [], []
    for rd in os.listdir(root_dir):
        path = os.path.join(root_dir, rd)
        if not (os.path.isdir(path) and rd.startswith('angle_dist_record_')):
            continue
        h5_files = glob.glob(os.path.join(path, '*.h5'))
        csv_files = glob.glob(os.path.join(path, '*.csv'))
        if not h5_files or not csv_files:
            continue
        with h5py.File(h5_files[0], 'r') as f:
            arr = f['DS1'][:].astype(np.float32)
        data = np.loadtxt(csv_files[0], delimiter=',', skiprows=1)
        angles = data[:, 1].astype(np.float32)
        dists = data[:, 2].astype(np.float32)
        xs_list.append(arr)
        angles_list.append(angles)
        dists_list.append(dists)
    return xs_list, angles_list, dists_list


# ---------- Model Definition ----------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))


class MultiHeadNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_ang = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            ResBlock(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128), nn.AdaptiveAvgPool2d(1),
        )
        self.backbone_dist = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            ResBlock(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            ResBlock(128), nn.AdaptiveAvgPool2d(1),
        )
        # 增加 bidirectional, 2-layer GRU
        self.gru_ang = nn.GRU(128, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.gru_dist = nn.GRU(128, 128, num_layers=2, bidirectional=True, batch_first=True)
        # head 輸出大小要考慮 bidirectional -> 128*2
        self.head_angle = nn.Sequential(nn.Linear(128*2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
        self.head_dist  = nn.Sequential(nn.Linear(128*2, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        B, T, C, H, W = x.shape
        xa = x.view(B*T, C, H, W)
        fa = self.backbone_ang(xa).view(B, T, -1)
        outa, _ = self.gru_ang(fa)
        a = self.head_angle(outa[:, -1]).squeeze(1)
        xd = x[:, :, 0:1, :, :].contiguous().view(B*T, 1, H, W)
        fd = self.backbone_dist(xd).view(B, T, -1)
        outd, _ = self.gru_dist(fd)
        d = self.head_dist(outd[:, -1]).squeeze(1)
        return a, d


# ---------- Training Pipeline ----------
def train_model(root='Record', seq_len=20, stride=1,
                epochs=100, batch_size=64, lr=1e-3,
                patience=10):
    # load raw sequences
    xs_list, angs_list, dists_list = load_all_data(root)
    # compute mean/std on raw frames (no duplication)
    all_raw = np.concatenate(xs_list, axis=0)
    mean = all_raw.mean(axis=(0, 2, 3), keepdims=True)
    std = all_raw.std(axis=(0, 2, 3), keepdims=True)
    print("Mean/std per channel:", mean.flatten(), std.flatten())

    # compute target scaling stats
    angs = np.concatenate(angs_list)
    dists = np.concatenate(dists_list)
    ang_mean, ang_std = angs.mean(), angs.std()
    dist_mean, dist_std = dists.mean(), dists.std()
    print(f"Angle mean/std: {ang_mean:.4f}/{ang_std:.4f}")
    print(f"Dist  mean/std: {dist_mean:.4f}/{dist_std:.4f}")

    # build sliding windows
    windows = []
    for xs, ys, ds in zip(xs_list, angs_list, dists_list):
        N = xs.shape[0]
        for i in range(0, N - seq_len + 1, stride):
            windows.append((xs[i:i+seq_len], (ys[i+seq_len-1], ds[i+seq_len-1])))

    # split
    idxs = np.random.permutation(len(windows))
    n = len(idxs)
    ntr = int(0.8 * n); nva = int(0.1 * n)
    tr_idx, va_idx = idxs[:ntr], idxs[ntr:ntr+nva]
    te_idx = idxs[ntr+nva:]
    train_w = [windows[i] for i in tr_idx]
    val_w   = [windows[i] for i in va_idx]
    test_w  = [windows[i] for i in te_idx]

    train_ds = WindowDataset(train_w, mean, std)
    val_ds   = WindowDataset(val_w,   mean, std)
    test_ds  = WindowDataset(test_w,  mean, std)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True,
                              num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False,
                             num_workers=10, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    model = MultiHeadNet().to(DEVICE)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    best_val = float('inf'); epochs_no_improve = 0
    train_losses, val_losses = [], []

    for ep in range(1, epochs+1):
        model.train(); tr_loss=0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            a_gt, d_gt = yb[:,0], yb[:,1]
            a_gt_n = (a_gt - ang_mean)/ang_std
            d_gt_n = (d_gt - dist_mean)/dist_std
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                a_pred, d_pred = model(xb)
                a_pred_n = (a_pred - ang_mean)/ang_std
                d_pred_n = (d_pred - dist_mean)/dist_std
                # weighted loss
                loss = 0.7 * criterion(a_pred_n, a_gt_n) + 0.3 * criterion(d_pred_n, d_gt_n)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer); scaler.update()
            tr_loss += loss.item()*xb.size(0)
        tr_loss /= len(train_ds); train_losses.append(tr_loss)

        model.eval(); val_loss=0
        with torch.no_grad():
            for xb,yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                a_gt, d_gt = yb[:,0], yb[:,1]
                a_gt_n = (a_gt - ang_mean)/ang_std
                d_gt_n = (d_gt - dist_mean)/dist_std
                with torch.cuda.amp.autocast():
                    a_pred, d_pred = model(xb)
                    a_pred_n = (a_pred - ang_mean)/ang_std
                    d_pred_n = (d_pred - dist_mean)/dist_std
                    val_loss += (0.7*criterion(a_pred_n,a_gt_n) + 0.3*criterion(d_pred_n,d_gt_n)).item()*xb.size(0)
        val_loss /= len(val_ds); val_losses.append(val_loss)
        print(f"Epoch {ep}/{epochs} | TrainL {tr_loss:.4f} | ValL {val_loss:.4f}")
        scheduler.step()
        if val_loss < best_val - 1e-4:
            best_val = val_loss; epochs_no_improve=0
            torch.save(model.state_dict(),'best_model.pth')
        else:
            epochs_no_improve +=1
            if epochs_no_improve>=patience:
                print(f"Early stopping at epoch {ep}."); break

    # load & test
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    all_at,all_dt,all_ap,all_dp = [],[],[],[]
    with torch.no_grad():
        for xb,yb in test_loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE)
            at,dt=yb[:,0].cpu().numpy(),yb[:,1].cpu().numpy()
            ap,dp=model(xb)
            ap, dp = ap.cpu().numpy(), dp.cpu().numpy()
            all_at.append(at); all_dt.append(dt)
            all_ap.append(ap); all_dp.append(dp)
    at,dt=np.concatenate(all_at), np.concatenate(all_dt)
    ap,dp=np.concatenate(all_ap), np.concatenate(all_dp)
    mae_a,rmse_a,r2_a,da = regression_metrics(at, ap)
    mae_d,rmse_d,r2_d,dd = regression_metrics(dt, dp)
    print(f"TEST ANGLE => MAE {mae_a:.2f}°, RMSE {rmse_a:.2f}°, R2 {r2_a:.3f}")
    print(f"TEST DIST  => MAE {mae_d:.2f}cm, RMSE {rmse_d:.2f}cm, R2 {r2_d:.3f}")

    # save metrics & plot
    with open('metrics.txt','w') as f:
        f.write(f"angle_MAE,{mae_a:.4f}\nangle_RMSE,{rmse_a:.4f}\nangle_R2,{r2_a:.4f}\n")
        for k,v in da.items(): f.write(f"angle_{k},{v:.4f}\n")
        f.write(f"dist_MAE,{mae_d:.4f}\ndist_RMSE,{rmse_d:.4f}\ndist_R2,{r2_d:.4f}\n")
        for k,v in dd.items(): f.write(f"dist_{k},{v:.4f}\n")
    if SAVE_PLOTS:
        plt.figure(); plt.plot(train_losses,label='Train'); plt.plot(val_losses,label='Val')
        plt.xlabel('Epoch'); plt.ylabel('Normalized Loss'); plt.legend(); plt.tight_layout()
        plt.savefig('loss_curve.png'); plt.close()
    torch.save(model.state_dict(),'final_model.pth')
    print("Training complete.")

if __name__ == '__main__':
    train_model()
