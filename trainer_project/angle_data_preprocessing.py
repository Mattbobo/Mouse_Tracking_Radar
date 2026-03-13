# -*- coding: utf-8 -*-
"""
Preprocess angle data into NPZ splits + mean/std text.

Output files (default):
- Process_data/Angle_data/train.npz  (X: [N,T,C,H,W], y: [N])
- Process_data/Angle_data/val.npz
- Process_data/Angle_data/test.npz
- Process_data/Angle_data/mean_std.txt   (two lines: mean, std)

You can run:
python Process_data/Angle_data/angle_data_preprocessing.py
"""
import os, glob, csv
import numpy as np
import h5py

# ========== Config ==========
DATA_ROOT = os.path.join('Record')               # 來源資料夾
OUT_DIR   = os.path.join('Process_data', 'Angle_data')
SEQ_LEN   = 20
STRIDE    = 1
VAL_RATIO = 0.10
TEST_RATIO= 0.10
SEED      = 42
ROUND_LABEL_ANGLE_1_DECIMAL = True  # 是否把 angle_deg 四捨五入到 0.1 度

os.makedirs(OUT_DIR, exist_ok=True)

def read_one_record_dir(rec_dir):
    """
    回傳 (xs, ys)
      xs: (N, T?, C, H, W) 未切窗的序列資料，實際是 (frames, C, H, W)
      ys: (N,) 每 frame 的角度標籤（之後取視窗最後一幀）
    """
    h5_list  = glob.glob(os.path.join(rec_dir, '*.h5'))
    csv_list = glob.glob(os.path.join(rec_dir, '*.csv'))
    if not h5_list or not csv_list:
        return None, None

    # 讀 radar DS1
    with h5py.File(h5_list[0], 'r') as f:
        xs = f['DS1'][:]  # shape: (frames, C, H, W)

    # 讀 angle_deg
    with open(csv_list[0], 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            idx = header.index('angle_deg')
        except ValueError:
            raise RuntimeError(f"'angle_deg' 欄位不在 {csv_list[0]} 的表頭中: {header}")
        ys = []
        for row in reader:
            ys.append(float(row[idx]))
        ys = np.asarray(ys, dtype=np.float32)

    # 對齊長度（偶爾 csv 或 h5 會差 1）
    L = min(xs.shape[0], ys.shape[0])
    xs, ys = xs[:L], ys[:L]

    if ROUND_LABEL_ANGLE_1_DECIMAL:
        ys = np.round(ys, 1)

    return xs.astype(np.float32), ys.astype(np.float32)

def build_windows(xs_list, ys_list, seq_len, stride):
    """將多段資料切成 (X_windows, y_last)"""
    Xw, Yw = [], []
    for xs, ys in zip(xs_list, ys_list):
        n = xs.shape[0]
        if n < seq_len:
            continue
        for i in range(0, n - seq_len + 1, stride):
            Xw.append(xs[i:i+seq_len])       # (T,C,H,W)
            Yw.append(ys[i + seq_len - 1])   # 視窗最後一幀的標籤
    if not Xw:
        raise RuntimeError("沒有切出任何視窗，請檢查 SEQ_LEN/STRIDE 與資料是否充足。")
    Xw = np.stack(Xw, axis=0).astype(np.float32)  # (N,T,C,H,W)
    Yw = np.asarray(Yw, dtype=np.float32)         # (N,)
    return Xw, Yw

def save_npz(path, X, y):
    np.savez_compressed(path, X=X, y=y)

def main():
    np.random.seed(SEED)

    # 1) 收集所有資料夾
    rec_dirs = [
        os.path.join(DATA_ROOT, d)
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('angle_dist_record_')
    ]
    if not rec_dirs:
        raise RuntimeError(f"在 {DATA_ROOT} 底下找不到 'angle_dist_record_*' 開頭的資料夾。")

    # 2) 逐一讀取
    xs_list, ys_list = [], []
    for rd in rec_dirs:
        xs, ys = read_one_record_dir(rd)
        if xs is None:
            continue
        xs_list.append(xs); ys_list.append(ys)

    # 3) 切窗
    X, y = build_windows(xs_list, ys_list, SEQ_LEN, STRIDE)  # X: (N,T,C,H,W)

    # 4) 統計 mean/std（對全部 X 的像素做）
    mean_all = X.mean()
    std_all  = X.std()
    if std_all <= 0:
        raise RuntimeError("std 計算為 0，請檢查資料。")

    # 5) 切分 train/val/test
    N = X.shape[0]
    idxs = np.random.permutation(N)
    n_val = max(int(VAL_RATIO * N), 1)
    n_test= max(int(TEST_RATIO * N), 1)
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise RuntimeError("訓練樣本數 <= 0，請調整 VAL_RATIO/TEST_RATIO。")

    tr_idx = idxs[:n_train]
    va_idx = idxs[n_train:n_train+n_val]
    te_idx = idxs[n_train+n_val:]

    save_npz(os.path.join(OUT_DIR, 'train.npz'), X[tr_idx], y[tr_idx])
    save_npz(os.path.join(OUT_DIR, 'val.npz'),   X[va_idx], y[va_idx])
    save_npz(os.path.join(OUT_DIR, 'test.npz'),  X[te_idx], y[te_idx])

    # 6) 存 mean/std
    with open(os.path.join(OUT_DIR, 'mean_std.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{mean_all:.8f}\n{std_all:.8f}\n")

    print(f"Done. Windows: {N}  | Train:{len(tr_idx)}  Val:{len(va_idx)}  Test:{len(te_idx)}")
    print(f"Mean: {mean_all:.6f}, Std: {std_all:.6f}")
    print(f"Saved to: {OUT_DIR}")

if __name__ == '__main__':
    main()
