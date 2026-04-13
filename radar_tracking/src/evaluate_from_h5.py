import numpy as np, h5py, csv, matplotlib.pyplot as plt, pickle, zmq
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# === ZeroMQ Client (連線到模型伺服器) ===
ctx = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")

def ask_angle(seq):
    sock.send(pickle.dumps(('angle', seq), protocol=4))
    angle, _ = sock.recv_pyobj()
    return float(angle)

def ask_dist(seq):
    sock.send(pickle.dumps(('dist', seq), protocol=4))
    _, dist = sock.recv_pyobj()
    return float(dist)

# === Normalization (與訓練一致) ===
mean_arr = np.array([22.960949, 27.205647], dtype=np.float32)
std_arr  = np.array([66.738190, 74.960556], dtype=np.float32)
SEQ_LEN = 20
FPS = 60.0   # 取樣頻率

# === 輸入檔案 ===
h5_path = r"C:\Users\mc2\Desktop\KKT_Module_Example_20240820\Record\Random_Test\data_20251013_141541.h5"
csv_path = r"C:\Users\mc2\Desktop\KKT_Module_Example_20240820\Record\Random_Test\records.csv"

# === 讀取資料 ===
with h5py.File(h5_path, 'r') as f:
    ds = np.array(f['DS1'])
N = ds.shape[0]
angles, dists = [], []
with open(csv_path) as f:
    next(f)
    for row in csv.reader(f):
        _, a, d = row
        angles.append(float(a))
        dists.append(float(d))
angles, dists = np.array(angles), np.array(dists)

# === 預測 ===
pred_a, pred_d = [], []
for i in range(SEQ_LEN - 1, N):
    seq = ds[i - SEQ_LEN + 1 : i + 1].astype(np.float32)
    seq_norm = (seq - mean_arr[None, :, None, None]) / std_arr[None, :, None, None]
    pred_a.append(ask_angle(seq_norm))
    pred_d.append(ask_dist(seq_norm[:, 0:1, :, :]))
pred_a, pred_d = np.array(pred_a), np.array(pred_d)

# === 對齊 Label ===
gt_a = angles[SEQ_LEN - 1 :]
gt_d = dists[SEQ_LEN - 1 :]
frames = np.arange(len(pred_a))
time_s = frames / FPS

# === 指標 ===
mae_a = mean_absolute_error(gt_a, pred_a)
mae_d = mean_absolute_error(gt_d, pred_d)
rmse_a = math.sqrt(mean_squared_error(gt_a, pred_a))
rmse_d = math.sqrt(mean_squared_error(gt_d, pred_d))
r2_a = r2_score(gt_a, pred_a)
r2_d = r2_score(gt_d, pred_d)
print(f"Angle: MAE={mae_a:.2f}°, RMSE={rmse_a:.2f}, R²={r2_a:.3f}")
print(f"Range: MAE={mae_d:.2f}cm, RMSE={rmse_d:.2f}, R²={r2_d:.3f}")

# === 繪圖 ===
plt.figure(figsize=(10,7.5))
plt.suptitle('Radar Model vs Ground Truth', fontsize=15, fontweight='bold')

# --- 上圖：Angle ---
ax1 = plt.subplot(2,1,1)
ax1.plot(time_s, gt_a, label='Ground Truth (Angle)', linewidth=1.8)
ax1.plot(time_s, pred_a, '--', label='Prediction (Angle)', linewidth=1.8)
ax1.set_ylabel('Angle (°)')
ax1.set_xlabel('Time (s)')
ax1.grid(True)
ax1.legend(loc='upper right', fontsize=10, frameon=True)
ax1.text(0.98, 0.05, f'MAE = {mae_a:.2f}°',
         transform=ax1.transAxes, ha='right', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# ✅ 固定角度刻度與範圍
ax1.set_ylim(-45, 45)
ax1.set_yticks(np.arange(-45, 46, 15))

# --- 下圖：Range ---
ax2 = plt.subplot(2,1,2)
ax2.plot(time_s, gt_d, label='Ground Truth (Range)', linewidth=1.8)
ax2.plot(time_s, pred_d, '--', label='Prediction (Range)', linewidth=1.8)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Range (cm)')
ax2.grid(True)
ax2.legend(loc='upper right', fontsize=10, frameon=True)
ax2.text(0.98, 0.05, f'MAE = {mae_d:.2f} cm',
         transform=ax2.transAxes, ha='right', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

# ✅ 固定距離刻度與範圍
ax2.set_ylim(0, 30)
ax2.set_yticks(np.arange(0, 31, 5))

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig('offline_eval_result.png', dpi=300, bbox_inches='tight')
plt.show()
