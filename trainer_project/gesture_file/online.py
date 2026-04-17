# -*- coding: utf-8 -*-
"""
online.py — Online mmWave Gesture Inference (CNN3D + Prototype + per-class distance thresholds)

需求重點：
- KKT 區塊（連線/Receiver/FRM/UI 模組）維持不變
- 每幀先做正規化 (RDI, PHD 各自 μ/σ)
- 滑動視窗 T=30，組成 (1,2,30,32,32) 餵 CNN3DBackbone 得到 embedding z
- 讀 out/prototypes.npz（讀法與 offline.py 相同），計 z 與每個 prototype 的距離（L2）
- 找最小距離所屬類別，並套用「每類別距離門檻」：come / patpat / wave 各一
- K-rule：confirm_k 連續同類非 0 進入；exit_k 連續不同退出
- 退出後維持顯示 hold_k 個視窗的最後手勢（避免畫面抖動）

Labels: 0=Background, 1=Come, 2=Wave, 3=PatPat（注意：以 prototypes.npz 的 gesture_classes 對齊）
"""

import os, sys, time, argparse
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from PySide2 import QtWidgets, QtCore

# ===== 你的模型（直接使用 CNN3DBackbone） =====
from model import CNN3DBackbone

# ===== KKT imports（保留不動） =====
from KKT_Module import kgl
from KKT_Module.DataReceive.Core import Results
from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
from KKT_Module.FiniteReceiverMachine import FRM
from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc
from KKT_Module.GuiUpdater.GuiUpdater import Updater

from KKT_UI.KKTGraph import ShowFeatureMap  # for RDI/PHD widget  (2x 32x32)


# ------------------- 預設參數（可用 CLI 覆蓋） -------------------
DEFAULT_MODEL_PATH     = "out/model_last.pth"
DEFAULT_PROTO_PATH     = "out/prototypes_last.npz"
DEFAULT_SETTING_FILE   = r"D:\Ben_radar\radar_1029\TempParam\K60168-Test-00256-008-v0.0.8-20230717_60cm"
DEFAULT_STREAM         = "feature_map"    # or "raw_data"
DEFAULT_WINDOW         = 20
DEFAULT_BATCH_MS_LOG   = 50               # 每 50 個 window 印一次平均 latency

# 正規化參數（逐幀套用；與 offline 對齊）
DEFAULT_RDI_MEAN = 17.61
DEFAULT_RDI_STD  = 56.18
DEFAULT_PHD_MEAN = 18.97
DEFAULT_PHD_STD  = 58.29

# 每類別距離門檻（手動給）
DEFAULT_THR_Gesture   = 0.7


# K-rule 與退出後維持顯示
DEFAULT_CONFIRM_K = 5
DEFAULT_EXIT_K    = 5
DEFAULT_HOLD_K    = 10
# ----------------------------------------------------------------


# ====================== KKT helpers（沿用） ======================
def connect_device():
    try:
        device = kgl.ksoclib.connectDevice()
        if device == 'Unknow':
            ret = QtWidgets.QMessageBox.warning(
                None, 'Unknown Device', 'Please reconnect device and try again',
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            )
            if ret == QtWidgets.QMessageBox.Ok:
                connect_device()
    except Exception:
        ret = QtWidgets.QMessageBox.warning(
            None, 'Connection Failed', 'Please reconnect device and try again',
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        if ret == QtWidgets.QMessageBox.Ok:
            connect_device()

def run_setting_script(setting_name: str):
    ksp = SettingProc()
    cfg = SettingConfigs()
    cfg.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
    cfg.Processes = [
        'Reset Device','Gen Process Script','Gen Param Dict','Get Gesture Dict',
        'Set Script','Run SIC','Phase Calibration','Modulation On'
    ]
    cfg.setScriptDir(f'{setting_name}')
    ksp.startUp(cfg)

def set_properties(obj: object, **kwargs):
    print(f"==== Set properties in {obj.__class__.__name__} ====")
    for k, v in kwargs.items():
        if not hasattr(obj, k):
            print(f'Attribute "{k}" not in {obj.__class__.__name__}.'); continue
        setattr(obj, k, v); print(f'Attribute "{k}", set "{v}"')
# =================================================================


# ================== 串流 K-rule + 退出後維持顯示 =================
class StreamingKRuleWithHold:
    """
    - 連續 >= confirm_k 個同類且非 0 => 進入該手勢
    - 進入後，連續 exit_k 個「不等於當前手勢」 => 退出（stable=0），但 GUI 顯示維持 last_gesture 共 hold_k 視窗
    - step(raw_label) -> (stable_label, display_label)
        stable_label：K-rule 穩定結果（0/1/2/3）
        display_label：考慮 hold_k 的最終顯示（0/1/2/3）
    """
    def __init__(self, confirm_k: int, exit_k: int, hold_k: int):
        self.confirm_k = int(confirm_k)
        self.exit_k = int(exit_k)
        self.hold_k = int(hold_k)

        # 進出狀態
        self.in_gesture = False
        self.cur_label = 0
        self.same_lab = None
        self.same_cnt = 0
        self.diff_cnt = 0

        # 退出後維持顯示
        self.hold_counter = 0
        self.last_gesture = 0

    def step(self, raw_label: int) -> Tuple[int, int]:
        # 追蹤連續相同 raw_label
        if raw_label == self.same_lab:
            self.same_cnt += 1
        else:
            self.same_lab, self.same_cnt = raw_label, 1

        # --- 尚未進入狀態 ---
        if not self.in_gesture:
            stable = 0
            # 進入條件：連續 confirm_k 且非 0
            if raw_label != 0 and self.same_cnt >= self.confirm_k:
                self.in_gesture = True
                self.cur_label = raw_label
                self.diff_cnt = 0
                self.hold_counter = 0
                self.last_gesture = self.cur_label
                stable = self.cur_label
            # 顯示：若還在 hold 期間，就顯示 hold；否則顯示 stable
            display = self._display_with_hold(stable)
            return stable, display

        # --- 已進入狀態 ---
        if raw_label == self.cur_label:
            self.diff_cnt = 0
            stable = self.cur_label
            self.last_gesture = self.cur_label
            self.hold_counter = 0
        else:
            self.diff_cnt += 1
            if self.diff_cnt >= self.exit_k:
                # 退出 → stable=0；啟動 hold
                self.in_gesture = False
                self.cur_label = 0
                self.diff_cnt = 0
                stable = 0
                # 啟動 hold
                if self.last_gesture != 0:
                    self.hold_counter = self.hold_k
            else:
                # 尚未退出，仍維持原手勢
                stable = self.cur_label

        display = self._display_with_hold(stable)
        return stable, display

    def _display_with_hold(self, stable_label: int) -> int:
        # 若 stable=0 且仍在 hold 期間，維持顯示 last_gesture
        if stable_label == 0 and self.hold_counter > 0:
            self.hold_counter -= 1
            return self.last_gesture
        return stable_label
# =================================================================


# ========================= 內建極簡 GUI ==========================
class MiniGestureGUI(QtWidgets.QWidget):
    """
    上：顯示「display_label」（已考慮 hold_k）
    下：RDI & PHD 即時影像
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture (Stable + Hold) + RDI/PHD")
        self.resize(900, 700)

        root = QtWidgets.QVBoxLayout(self)

        # ==== 上：手勢文字 ====
        self.lbl = QtWidgets.QLabel("gesture: Background")
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl.setStyleSheet(
            "font-size: 28px; font-weight: bold; padding: 12px; "
            "border-radius: 8px; background: #eeeeee;"
        )
        root.addWidget(self.lbl, stretch=0)

        # ==== 下：RDI/PHD 圖 ====
        self.fm_widget = ShowFeatureMap.MultiFeatureMapPlotsWidget()
        fm_wrap = QtWidgets.QFrame()
        fm_wrap.setFrameShape(QtWidgets.QFrame.StyledPanel)
        fm_wrap.setLayout(QtWidgets.QVBoxLayout())
        fm_wrap.layout().setContentsMargins(8, 8, 8, 8)
        fm_wrap.layout().addWidget(self.fm_widget)
        root.addWidget(fm_wrap, stretch=1)

    @QtCore.Slot(int)
    def set_gesture(self, label_for_display: int):
        names = {0: "Background", 1: "Gesture"}
        name = names.get(label_for_display, "Background")
        color = {
            "Background": "#eeeeee",
            "Gesture":       "#E6E6FA",
         
        }[name]
        self.lbl.setText(f"gesture: {name}")
        self.lbl.setStyleSheet(
            f"font-size: 28px; font-weight: bold; padding: 12px; "
            f"border-radius: 8px; background: {color};"
        )

    @QtCore.Slot(object)
    def set_feature_map(self, feature_map_arr):
        """feature_map_arr: shape (2,32,32) 或 (32,32,2)"""
        self.fm_widget.setData(feature_map_arr)
# =================================================================


# ========================= Online 核心 ===========================
def _normalize_frame(frame: np.ndarray, rdi_mean, rdi_std, phd_mean, phd_std) -> np.ndarray:
    """逐幀正規化；frame shape 可為 (2,32,32) 或 (32,32,2)，回傳 (2,32,32)"""
    x = np.asarray(frame)
    if x.shape == (32, 32, 2):
        x = np.transpose(x, (2, 0, 1))
    elif x.shape != (2, 32, 32):
        raise ValueError(f"Unexpected frame shape: {x.shape}")

    x = x.astype(np.float32, copy=True)
    x[0] = (x[0] - rdi_mean) / max(rdi_std, 1e-6)
    x[1] = (x[1] - phd_mean) / max(phd_std, 1e-6)
    return x


class OnlineContext:
    """
    滑動視窗推論（與 offline.py 同步的距離計算），再接 K-rule + hold_k
    """
    def __init__(
        self,
        model: CNN3DBackbone,
        prototypes: torch.Tensor,
        proto_labels: list,
        device: torch.device,
        window_size: int,
        rdi_mean: float, rdi_std: float, phd_mean: float, phd_std: float,
        thr_gesture: float,
        confirm_k: int, exit_k: int, hold_k: int,
        ms_log_interval: int = 50,
    ):
        self.model = model
        self.device = device

        self.prototypes = prototypes.to(device)           # (K,D)
        self.proto_labels = list(proto_labels)            # proto_idx -> true class {0,1,2,3}

        self.window = int(window_size)
        self.buffer = np.zeros((2, 32, 32, self.window), dtype=np.float32)
        self.collected = 0

        # per-class thresholds
        self.thr_gesture   = float(thr_gesture)

        # normalization params
        self.rdi_mean = float(rdi_mean)
        self.rdi_std  = float(rdi_std)
        self.phd_mean = float(phd_mean)
        self.phd_std  = float(phd_std)

        # k-rule + hold
        self.decider = StreamingKRuleWithHold(confirm_k=confirm_k, exit_k=exit_k, hold_k=hold_k)

        # perf
        self.n_forward = 0
        self.total_ms = 0.0
        self.ms_log_interval = int(ms_log_interval)

    @staticmethod
    def _dist_l2(z: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # z: (B,D), prototypes: (K,D) -> (B,K)
        return torch.cdist(z, prototypes)

    def _infer_one_window(self) -> Tuple[int, float, np.ndarray]:
        """
        回傳：
          raw_label (0/1/2/3，已套用每類別距離門檻)
          d_min     （到最近 prototype 的距離）
          all_dists （到每個 prototype 的距離，shape=(K,)）
        """
        # (C,H,W,T) -> (1,2,30,32,32)
        win = np.transpose(self.buffer, (0, 3, 1, 2))
        x = torch.from_numpy(win[None, ...]).float().to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            z    = self.model(x)                      # (1, D)
            dist = torch.cdist(z, self.prototypes)    # (1, K=3)
            d_min, k = torch.min(dist, dim=1)
            d_min_val = float(d_min.item())
            c_hat = int(self.proto_labels[int(k.item())])  # 1/2/3

            raw_label = 0
            if c_hat == 1 and d_min_val < self.thr_gesture:     raw_label = 1

        all_d = dist.squeeze(0).detach().cpu().numpy()      
        return raw_label, d_min_val, all_d                

    def push_and_step(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, float, np.ndarray]]:
        """
        塞一幀，當滿視窗時：
          回傳 (raw_label, stable_label, display_label, d_min, all_dists[K])
        未滿視窗回傳 None
        """
        # 逐幀正規化後進 buffer
        f = _normalize_frame(frame, self.rdi_mean, self.rdi_std, self.phd_mean, self.phd_std)
        self.buffer = np.roll(self.buffer, shift=-1, axis=-1)
        self.buffer[..., -1] = f
        self.collected += 1
        if self.collected < self.window:
            return None

        raw_label, d_min, all_dists = self._infer_one_window()
        stable_label, display_label = self.decider.step(raw_label)
        return raw_label, stable_label, display_label, d_min, all_dists
# =================================================================


class GuiBridge(QtCore.QObject):
    displayGestureChanged = QtCore.Signal(int)   # 顯示（含 hold）
    featureMapUpdated = QtCore.Signal(object)


# ============== Updater（繼承 Updater；透過 bridge 發訊號） ==============
class InferenceUpdater(Updater):
    def __init__(self, ctx: OnlineContext, stream: str = "feature_map", bridge: Optional[GuiBridge] = None):
        super().__init__()
        self.ctx = ctx
        self.stream = stream
        self.bridge = bridge
        self.last_display = -1  # -1 確保第一次會觸發更新

    def update(self, res: Results):
        try:
            # 來源：feature_map 或 raw_data
            arr = res['raw_data'].data if self.stream == "raw_data" else res['feature_map'].data
            out = self.ctx.push_and_step(arr)
            if out is None:
                return

            raw_label, stable_label, display_label, d_min, all_dists = out

            # 1) GUI 更新顯示（含 hold 邏輯）
            if display_label != self.last_display:
                self.last_display = display_label
                if self.bridge is not None:
                    self.bridge.displayGestureChanged.emit(int(display_label))

            # 2) Console 列印（含 raw / stable / display 與距離）
            name_map = {0: "Background", 1: "Gesture"}
            print(f"[Online] raw={name_map[raw_label]} | stable={name_map[stable_label]} | display={name_map[display_label]} | d_min={d_min:.4f}")

            # 3) GUI 更新 RDI/PHD
            try:
                fm = res['feature_map'].data
                if self.bridge is not None and fm is not None:
                    self.bridge.featureMapUpdated.emit(fm)
            except Exception:
                pass

        except Exception:
            # 靜默忽略單幀錯誤，避免中斷串流
            pass
# =================================================================


# =============================== 主程式 ===============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--prototype_path", type=str, default=DEFAULT_PROTO_PATH)
    ap.add_argument("--setting_file", type=str, default=DEFAULT_SETTING_FILE)
    ap.add_argument("--stream", type=str, default=DEFAULT_STREAM, choices=["feature_map","raw_data"])
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)

    # normalization
    ap.add_argument("--rdi_mean", type=float, default=DEFAULT_RDI_MEAN)
    ap.add_argument("--rdi_std",  type=float, default=DEFAULT_RDI_STD)
    ap.add_argument("--phd_mean", type=float, default=DEFAULT_PHD_MEAN)
    ap.add_argument("--phd_std",  type=float, default=DEFAULT_PHD_STD)

    # per-class distance thresholds
    ap.add_argument("--thr_gesture",   type=float, default=DEFAULT_THR_Gesture,   help="distance threshold for class 1 (Gesture)")

    # k-rule + hold
    ap.add_argument("--confirm_k", type=int, default=DEFAULT_CONFIRM_K, help="enter gesture when >=k consecutive same non-zero")
    ap.add_argument("--exit_k",    type=int, default=DEFAULT_EXIT_K,    help="exit gesture when >=k consecutive labels different")
    ap.add_argument("--hold_k",    type=int, default=DEFAULT_HOLD_K,    help="after exit, keep displaying last gesture for k windows")

    ap.add_argument("--ms_log_interval", type=int, default=DEFAULT_BATCH_MS_LOG)
    args = ap.parse_args()

    # 0) Qt 事件圈 + 內建 GUI
    app = QtWidgets.QApplication(sys.argv)
    gui = MiniGestureGUI()
    gui.show()

    # 1) 初始化雷達（KKT 保留不動）
    kgl.setLib()
    connect_device()
    run_setting_script(args.setting_file)

    # 切換輸出源
    if args.stream == "raw_data":
        kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
    else:
        kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)

    # 2) 載入模型（CNN3DBackbone）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN3DBackbone(in_channels=2, emb_dim=32).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[INFO] model loaded: {args.model_path} | device: {device}")

    # 3) 載入 prototypes（讀法與 offline.py 對齊）
    proto_npz   = np.load(args.prototype_path)
    prototypes  = torch.tensor(proto_npz["prototypes"], dtype=torch.float32)  # (3, D)
    proto_labels = [1]  # 依序對應 come, wave, patpat
    print(f"[INFO] Loaded prototypes: {prototypes.shape} | proto_labels={proto_labels}")

    # 建立 proto_labels（align gesture_classes）
    K = prototypes.shape[0]

    # 先預設正確對應（新檔沒寫 gesture_classes 時用這個）
    proto_labels = [1][:K]

    if "gesture_classes" in proto_npz:
        gc = list(np.array(proto_npz["gesture_classes"]).tolist())
        if len(gc) == K:
            proto_labels = gc
        elif len(gc) == K - 1:
            # 舊版含背景原型：假設 prototype[0] 是背景
            proto_labels = [0] + gc
        else:
            print(f"[WARN] gesture_classes length={len(gc)} mismatches K={K}; fallback to {proto_labels}")

    print(f"[INFO] Loaded prototypes: {prototypes.shape} | proto_labels={proto_labels}")


    # 4) 上線推論 context + bridge + updater
    ctx = OnlineContext(
        model=model,
        prototypes=prototypes,
        proto_labels=proto_labels,
        device=device,
        window_size=args.window,
        rdi_mean=args.rdi_mean, rdi_std=args.rdi_std,
        phd_mean=args.phd_mean, phd_std=args.phd_std,
        thr_gesture=args.thr_gesture,
        confirm_k=args.confirm_k, exit_k=args.exit_k, hold_k=args.hold_k,
        ms_log_interval=args.ms_log_interval,
    )
    bridge = GuiBridge()
    bridge.displayGestureChanged.connect(gui.set_gesture)
    bridge.featureMapUpdated.connect(gui.set_feature_map)

    updater = InferenceUpdater(ctx, stream=args.stream, bridge=bridge)

    # 5) Receiver + FRM
    receiver = MultiResult4168BReceiver()
    set_properties(receiver, actions=1, rbank_ch_enable=7, read_interrupt=0, clear_interrupt=0)
    FRM.setReceiver(receiver)
    FRM.setUpdater(updater)
    FRM.trigger()
    FRM.start()

    print("[INFO] Online inference started. Press Ctrl+C to quit.")
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        try: FRM.stop()
        except Exception: pass
        try: kgl.ksoclib.closeDevice()
        except Exception: pass
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
