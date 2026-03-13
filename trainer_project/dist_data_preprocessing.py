# -*- coding: utf-8 -*-
"""
Preprocess distance data (channel 0) into NPZ splits + mean/std (from TRAIN only).

Outputs:
- Process_data/Distance_data/train.npz  (X:[N,T,1,H,W], y:[N])
- Process_data/Distance_data/val.npz
- Process_data/Distance_data/test.npz
- Process_data/Distance_data/mean_std.txt  # line1: mean, line2: std (from train only)

Run:
python Process_data/Distance_data/distance_data_preprocessing.py
"""
import os, glob, csv
import numpy as np
import h5py

# ===== Config =====
DATA_ROOT = 'Record'
OUT_DIR   = os.path.join('Process_data', 'Distance_data')
SEQ_LEN   = 20
STRIDE    = 1
VAL_RATIO = 0.10
TEST_RATIO= 0.10
SEED      = 42
ROUND_LABEL_1_DECIMAL = True  # 距離是否四捨五入到 0.1 cm

os.makedirs(OUT_DIR, exist_ok=True)

def read_one_record_dir(rec_dir):
    """return (xs0, ds) where xs0:(N,1,H,W), ds:(N,)"""
    h5_list  = glob.glob(os.path.join(rec_dir, '*.h5'))
    csv_list = glob.glob(os.path.join(rec_dir, '*.csv'))
    if not h5_list or not csv_list:
        return None, None

    with h5py.File(h5_list[0], 'r') as f:
        arr = f['DS1'][:].astype(np.float32)     # (N,C,H,W)
    xs0 = arr[:, 0:1, :, :]                      # (N,1,H,W)

    # 讀距離欄位
    with open(csv_list[0], 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            idx = header.index('distance_cm')
        except ValueError:
            raise RuntimeError(f"'distance_cm' 欄位不在 {csv_list[0]} 表頭中: {header}")
        ds = []
        for row in reader:
            ds.append(float(row[idx]))
        ds = np.asarray(ds, dtype=np.float32)

    L = min(xs0.shape[0], ds.shape[0])
    xs0, ds = xs0[:L], ds[:L]
    if ROUND_LABEL_1_DECIMAL:
        ds = np.round(ds, 1)
    return xs0, ds

def build_windows(xs_list, ds_list, seq_len, stride):
    """return X:(N,T,1,H,W), y:(N,) from multiple sequences"""
    Xw, Yw = [], []
    for xs, ds in zip(xs_list, ds_list):
        n = xs.shape[0]
        if n < seq_len:
            continue
        for i in range(0, n - seq_len + 1, stride):
            Xw.append(xs[i:i+seq_len])          # (T,1,H,W)
            Yw.append(ds[i + seq_len - 1])      # label = 視窗最後一幀
    if not Xw:
        raise RuntimeError("沒有切出任何視窗；請檢查 SEQ_LEN/STRIDE 與資料。")
    Xw = np.stack(Xw, axis=0).astype(np.float32)  # (N,T,1,H,W)
    Yw = np.asarray(Yw, dtype=np.float32)         # (N,)
    return Xw, Yw

def save_npz(path, X, y):
    np.savez_compressed(path, X=X, y=y)

def main():
    np.random.seed(SEED)

    rec_dirs = [
        os.path.join(DATA_ROOT, d)
        for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith('angle_dist_record_')
    ]
    if not rec_dirs:
        raise RuntimeError(f"在 {DATA_ROOT} 找不到 'angle_dist_record_*'。")

    xs_list, ds_list = [], []
    for rd in rec_dirs:
        xs0, ds = read_one_record_dir(rd)
        if xs0 is None:
            continue
        xs_list.append(xs0); ds_list.append(ds)

    X, y = build_windows(xs_list, ds_list, SEQ_LEN, STRIDE)   # (N,T,1,H,W), (N,)

    # ---- split ----
    N = X.shape[0]
    idxs = np.random.permutation(N)
    n_val = max(int(VAL_RATIO * N), 1)
    n_test= max(int(TEST_RATIO * N), 1)
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise RuntimeError("訓練樣本數 <= 0；請調整 VAL/TEST 比例。")

    tr_idx = idxs[:n_train]
    va_idx = idxs[n_train:n_train+n_val]
    te_idx = idxs[n_train+n_val:]

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    # ---- mean/std from TRAIN ONLY ----
    mean_tr = X_tr.mean()
    std_tr  = X_tr.std()
    if std_tr <= 0:
        raise RuntimeError("std 計算為 0，請檢查資料。")

    save_npz(os.path.join(OUT_DIR, 'train.npz'), X_tr, y_tr)
    save_npz(os.path.join(OUT_DIR, 'val.npz'),   X_va, y_va)
    save_npz(os.path.join(OUT_DIR, 'test.npz'),  X_te, y_te)

    with open(os.path.join(OUT_DIR, 'mean_std.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{mean_tr:.8f}\n{std_tr:.8f}\n")

    print(f"Done. Windows: {N}  | Train:{len(tr_idx)}  Val:{len(va_idx)}  Test:{len(te_idx)}")
    print(f"[TRAIN] mean: {mean_tr:.6f}, std: {std_tr:.6f}")
    print(f"Saved to: {OUT_DIR}")

if __name__ == '__main__':
    main()
