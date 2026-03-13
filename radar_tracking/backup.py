import sys
import math
import pickle
import zmq
import numpy as np
from collections import deque
from PySide2 import QtWidgets, QtCore, QtGui
from KKT_Module import kgl
from KKT_Module.DataReceive.Core import Results
from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
from KKT_Module.FiniteReceiverMachine import FRM
from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc
from KKT_Module.GuiUpdater.GuiUpdater import Updater

# === Per‐channel normalization parameters from training ===
# channel 0: distance, channel 1: angle
mean_arr = np.array([22.960949, 27.205647], dtype=np.float32)
std_arr  = np.array([66.738190,  74.960556], dtype=np.float32)

# === ZeroMQ Client Setup ===
ctx  = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")

def ask_angle(seq: np.ndarray):
    """Send angle request, return angle (deg)."""
    sock.send(pickle.dumps(('angle', seq), protocol=4))
    angle, _ = sock.recv_pyobj()
    return angle

def ask_dist(seq: np.ndarray):
    """Send distance request, return distance (cm)."""
    sock.send(pickle.dumps(('dist', seq), protocol=4))
    _, dist = sock.recv_pyobj()
    return dist

# ========== Qt Signal / Updater ==========
class ShowDataSentinel(QtCore.QObject):
    sig_update_plot = QtCore.Signal(object)

class ShowDataViewModel(Updater):
    def __init__(self):
        super().__init__()
        self.sentinel = ShowDataSentinel()
    def update(self, res: Results):
        self.sentinel.sig_update_plot.emit(res)

# ========== Main GUI Widget ==========
class ShowDataView(QtWidgets.QWidget):
    def __init__(self, vm):
        super().__init__()
        self.vm = vm
        vm.sentinel.sig_update_plot.connect(self.updateRadar)

        # Sequence buffer
        self.seq_len = 20
        self.buffer  = deque(maxlen=self.seq_len)

        # Predictions (polar)
        self.pred_angle  = None   # deg
        self.pred_dist   = None   # cm

        # UI elements
        font = QtGui.QFont();
        font.setPointSize(14);
        font.setBold(True)

        self.pred_label = QtWidgets.QLabel('Predicted: --°, -- cm')
        self.pred_label.setFont(font)

        # ↓↓↓ 新增這兩個 Label 再設定字型 ↓↓↓
        self.pred_xy_label = QtWidgets.QLabel('Pred XY: --, -- cm')
        self.pred_xy_label.setFont(QtGui.QFont('', 12))

        self.sector_label = QtWidgets.QLabel()  # 繪 XY 面板
        self.sector_label.setFixedSize(600, 600)

        self.legend_label = QtWidgets.QLabel('🔴 Model Prediction')
        self.legend_label.setFont(QtGui.QFont('', 12))


        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.pred_label)
        hl.addStretch()
        layout.addLayout(hl)
        hl2 = QtWidgets.QHBoxLayout()
        hl2.addWidget(self.pred_xy_label)
        hl2.addStretch()
        layout.addLayout(hl2)
        layout.addWidget(self.sector_label, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.legend_label, alignment=QtCore.Qt.AlignHCenter)

        # ===== Drawing/Mapping params (XY Panel) =====
        # ===== XY Panel params (Pad in sector) =====
        # 扇形：±45°、Rmax=30cm
        self.A_deg = 45.0
        self.Rmax_cm = 30.0
        self.Xmax_cm = self.Rmax_cm * math.sin(math.radians(self.A_deg))  # 30*sin45 ≈ 21.2
        self.Ymax_cm = self.Rmax_cm

        # ---「扇形中間挖一塊矩形 Pad」---
        # 你可以改這三個：矩形高度、想要的寬度、以及距離雷達的起始 y0
        self.PAD_Y0_CM = 12.0  # 距雷達最近 10cm 起算（矩形底邊）
        self.PAD_H_CM = 16.0  # 矩形高度（沿距離軸 y）
        self.PAD_W_CM_DESIRED = 20.0  # 矩形寬度（左右 x，中心在 x=0）

        # 夾到扇形半徑內
        self.PAD_Y0_CM = max(0.0, min(self.PAD_Y0_CM, self.Rmax_cm))
        self.PAD_H_CM = max(1.0, min(self.PAD_H_CM, self.Rmax_cm - self.PAD_Y0_CM))

        # 為了讓整個矩形完全落在扇形內，寬度必須受「最靠近的那一條扇形邊」限制
        # 在距離 y 處，扇形半寬 = y * tan(A)
        tanA = math.tan(math.radians(self.A_deg))
        allowed_half_w_near = (self.PAD_Y0_CM) * tanA
        allowed_half_w_far = (self.PAD_Y0_CM + self.PAD_H_CM) * tanA
        allowed_half_w = min(allowed_half_w_near, allowed_half_w_far, self.Xmax_cm)

        self.PAD_HALF_W_CM = min(self.PAD_W_CM_DESIRED / 2.0, allowed_half_w)
        self.ACT_W_CM = 2.0 * self.PAD_HALF_W_CM  # Pad 實際寬
        self.ACT_H_CM = self.PAD_H_CM  # Pad 實際高
        # Pad 的「本地座標」定義：x ∈ [-ACT_W/2, +ACT_W/2], y ∈ [0, ACT_H]
        # （注意：全域距離是 y_global = PAD_Y0 + y_local）

        # ---- Adaptive gating (EMA + hysteresis) ----
        self.ema_energy = None  # 雜訊底 (會自動學習)
        self.ema_alpha_energy = 0.2  # EMA 係數，越小越穩
        self.on_snr = 1.5  # 進入「有目標」的 SNR 門檻
        self.off_snr = 1.2  # 離開「有目標」的 SNR 門檻
        self.min_frames_on = 2  # 連續幾幀達到 on_snr 才算偵測到
        self.min_frames_off = 3  # 連續幾幀低於 off_snr 才算消失
        self._detected_state = False  # 目前狀態
        self._on_cnt = 0
        self._off_cnt = 0

        # ---- Fixed absolute-energy gating (simple + hysteresis) ----
        self.ENERGY_ON = 15000.0  # 總能量「出現目標」門檻
        self.ENERGY_OFF = 12000.0  # 總能量「消失目標」門檻（略低一些形成遲滯）
        self._detected_abs = False  # 目前是否視為「有目標」


        # 畫點大小與軌跡/平滑
        self.point_size = 15
        self.ema_alpha_xy = 0.3
        self.pred_xy_ema = None

        # === Grid 與「掉門檻時保留上一點」 ===
        self.grid_cm = 2.0  # 每格 2 cm（正方形）
        self.pred_hold_local = None  # (x_local, y_local) 以 cm 為單位
        self.pred_hold_active = False  # True 代表用「保留點」顯示紅點

        self.deadband_cm = 2.0  # 停留死區半徑（小於此就視為沒動）
        self.unlock_cm = 4.0  # 解鎖半徑（移出這麼多才解除鎖定）
        self.still_need_frames = 5  # 連續幾幀算真的停住
        self._still_cnt = 0
        self._lock_still = False

    def _inside_pad(self, x_local: float, y_local: float) -> bool:
        return (-self.ACT_W_CM / 2 <= x_local <= self.ACT_W_CM / 2) and (0.0 <= y_local <= self.ACT_H_CM)

    # ---- Helpers: polar -> xy (cm) -> canvas px ----
    def polar_to_xy_cm(self, angle_deg: float, dist_cm: float):
        rad = math.radians(angle_deg)
        x = dist_cm * math.sin(rad)  # 右正
        y = dist_cm * math.cos(rad)  # 前正
        return x, y

    def xy_cm_to_px(self, x_cm: float, y_cm: float, w: int, h: int):
        # 以 FOV 框等比映射到正方形畫布
        sx = w / (2 * self.Xmax_cm)
        sy = h / (self.Ymax_cm)
        s  = min(sx, sy)  # 等比縮放
        px = w/2 + s * x_cm
        py = h   - (s * y_cm)
        return int(px), int(py)

    def _update_gate(self, total_energy: float) -> bool:
        # 初始化 baseline
        if self.ema_energy is None:
            self.ema_energy = total_energy if total_energy > 0 else 1.0

        # 計算 SNR
        snr = total_energy / max(self.ema_energy, 1e-6)

        # 僅在「未偵測」狀態下更新 baseline，避免被目標拉高
        if not self._detected_state:
            self.ema_energy = (1 - self.ema_alpha_energy) * self.ema_energy + self.ema_alpha_energy * total_energy

        # 遲滯＋多幀確認
        if self._detected_state:
            if snr <= self.off_snr:
                self._off_cnt += 1
                if self._off_cnt >= self.min_frames_off:
                    self._detected_state = False
                    self._off_cnt = 0
                    self._on_cnt = 0
            else:
                self._off_cnt = 0
        else:
            if snr >= self.on_snr:
                self._on_cnt += 1
                if self._on_cnt >= self.min_frames_on:
                    self._detected_state = True
                    self._on_cnt = 0
                    self._off_cnt = 0
            else:
                self._on_cnt = 0

        return self._detected_state

    def _update_gate_abs(self, total_energy: float) -> bool:
        # 簡單遲滯：高於 ON → True；低於 OFF → False；中間 → 維持
        if total_energy >= self.ENERGY_ON:
            self._detected_abs = True
        elif total_energy <= self.ENERGY_OFF:
            self._detected_abs = False
        return self._detected_abs

    def updateRadar(self, res: Results):
        # get radar data
        arr = np.array(
            res['feature_map'].data
            if hasattr(res, 'feature_map')
            else res['raw_data'].data,
            dtype=np.float32
        )
        # === 絕對能量 gating（只決定是否顯示，不中斷流程）===
        total_energy = float(np.sum(np.abs(arr)))
        detected = self._update_gate_abs(total_energy)

        proc = np.log10(arr + 1e-6) if arr.ndim == 2 else arr
        if proc.ndim == 2:
            proc = np.stack([proc, proc], 0)  # make 2-channel

        # push to buffer（即使未偵測也維持序列長度）
        self.buffer.append(proc)
        if len(self.buffer) < self.seq_len:
            # 序列尚未湊滿，先清空紅點顯示
            self.pred_angle = None
            self.pred_dist = None
            self.pred_xy_ema = None
            self.pred_label.setText('Predicted: --°, -- cm')
            self.pred_xy_label.setText('Pred Grid XY: --, -- (cells)')
            self.updateXY()
            return

        # build sequence & normalize
        seq = np.stack(self.buffer, axis=0).astype(np.float32)
        seq_norm = (seq - mean_arr[None, :, None, None]) / std_arr[None, :, None, None]

        # run models（即使未偵測也先算好）
        angle = ask_angle(seq_norm)  # deg
        seq_dist = seq_norm[:, 0:1, :, :]  # (T,1,H,W)
        dist = ask_dist(seq_dist)  # cm

        # 先把本幀的原始(未夾限)換成 Pad 本地座標（cm）
        gx, gy = self.polar_to_xy_cm(float(angle), float(dist))
        x_local_raw = gx
        y_local_raw = gy - self.PAD_Y0_CM

        cell = self.grid_cm  # 每格大小（2 cm）

        if not detected:
            if self.pred_hold_local is not None and self._inside_pad(*self.pred_hold_local):
                # 保留上一個紅點（凍結顯示）
                self.pred_hold_active = True
                self.pred_angle = None
                self.pred_dist = None
                hx, hy = self.pred_hold_local
                self.pred_label.setText('Predicted: --°, -- cm')
                self.pred_xy_label.setText(f'Pred Grid XY: {hx / self.grid_cm:.1f}, {hy / self.grid_cm:.1f} (cells)')
                self.updateXY()
                return

            # 沒歷史點或歷史點在範圍外 → 清掉
            self.pred_hold_active = False
            self.pred_hold_local = None
            self.pred_angle = None
            self.pred_dist = None
            self.pred_xy_ema = None
            self.pred_label.setText('Predicted: --°, -- cm')
            self.pred_xy_label.setText('Pred Grid XY: --, -- (cells)')
            self.updateXY()
            return

        # ---- 門檻通過 → 正常更新顯示，並記錄上一點（供掉門檻時保留）----
        self.pred_hold_active = False
        self.pred_angle, self.pred_dist = angle, dist
        self.pred_label.setText(f'Predicted: {angle:.1f}°, {dist:.1f} cm')

        # 先做「未夾限」本地座標 + 翻轉（注意：**不要**先夾邊界）
        x_u = x_local_raw
        y_u = y_local_raw
        x_u = -x_u
        y_u = self.ACT_H_CM - y_u

        # 若未夾限位置已不在範圍 → 直接清掉並返回（不畫、不保留）
        if not self._inside_pad(x_u, y_u):
            self.pred_hold_active = False
            self.pred_hold_local = None
            self.pred_angle = None
            self.pred_dist = None
            self.pred_xy_ema = None
            self.pred_label.setText('Predicted: --°, -- cm')
            self.pred_xy_label.setText('Pred Grid XY: --, -- (cells)')
            self.updateXY()
            return

        # 在範圍內 → 慢速鎖定/死區處理
        if self.pred_hold_local is not None:
            last_x, last_y = self.pred_hold_local
            d_now = math.hypot(x_u - last_x, y_u - last_y)

            # 死區：幾乎沒動就累加 still 計數
            if d_now < self.deadband_cm:
                self._still_cnt += 1
            else:
                self._still_cnt = 0

            # 進入鎖定
            if self._still_cnt >= self.still_need_frames:
                self._lock_still = True

            # 鎖定時：不到解鎖距離就沿用舊點；超過才解鎖
            if self._lock_still and d_now < self.unlock_cm:
                x_u, y_u = last_x, last_y
            elif self._lock_still and d_now >= self.unlock_cm:
                self._lock_still = False
                self._still_cnt = 0

        # 顯示前最後一步才夾邊界（避免畫出界，但不影響「是否在範圍」的邏輯）
        x_local = max(-self.ACT_W_CM / 2, min(self.ACT_W_CM / 2, x_u))
        y_local = max(0.0, min(self.ACT_H_CM, y_u))

        self.pred_hold_local = (x_local, y_local)
        self.pred_xy_label.setText(f'Pred Grid XY: {x_local / cell:.1f}, {y_local / cell:.1f} (cells)')
        self.updateXY()

    # ===== 核心：把扇形改成方形 XY 顯示 =====
    def updateXY(self):
        w, h = self.sector_label.width(), self.sector_label.height()
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.black)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # 外框
        p.setPen(QtGui.QPen(QtCore.Qt.white, 2))
        p.drawRect(0, 0, w - 1, h - 1)

        # === 以「格子數量」決定像素 ===
        margin = 20
        cell = self.grid_cm  # 每格 2 cm
        NX = int(round(self.ACT_W_CM / cell))  # 水平格數
        NY = int(round(self.ACT_H_CM / cell))  # 垂直格數

        s = min((w - 2 * margin) / NX, (h - 2 * margin) / NY)  # 每格的像素邊長（正方形）
        rect_w = int(round(NX * s))
        rect_h = int(round(NY * s))

        rect_left = int(round(w / 2 - rect_w / 2))
        rect_right = rect_left + rect_w
        rect_bottom = int(h - margin)  # 本地 y=0 在這條線
        rect_top = rect_bottom - rect_h  # 本地 y=ACT_H 在這條線

        # Pad 矩形
        p.setPen(QtGui.QPen(QtCore.Qt.white, 3))
        p.drawRect(QtCore.QRect(rect_left, rect_top, rect_w, rect_h))

        # 格線（每 1 格 = 2 cm）
        p.setPen(QtGui.QPen(QtCore.Qt.gray, 1))
        # 水平格線
        for i in range(1, NY):
            y = rect_bottom - int(round(i * s))
            p.drawLine(rect_left, y, rect_right, y)
        # 垂直格線
        for j in range(1, NX):
            x = rect_left + int(round(j * s))
            p.drawLine(x, rect_top, x, rect_bottom)

        # cm -> px（用「格數」來換算）
        def local_cm_to_px(x_local: float, y_local: float):
            # 夾到 Pad 邊界
            x_local = max(-self.ACT_W_CM / 2, min(self.ACT_W_CM / 2, x_local))
            y_local = max(0.0, min(self.ACT_H_CM, y_local))
            # 先換成「格數」
            gx = x_local / cell  # 例如 x_local = +6cm、cell=2 → gx=+3 格
            gy = y_local / cell  # y_local = 4cm → gy=2 格
            px = int(round((rect_left + rect_right) / 2 + gx * s))
            py = int(round(rect_bottom - gy * s))
            return px, py

        # 畫點（gating 通過才會有 pred_angle/dist）
        def draw_point(angle, dist, color, ema_xy_attr):
            if angle is None or dist is None:
                return
            angle = max(-self.A_deg, min(self.A_deg, float(angle)))
            dist = max(0.0, min(self.Rmax_cm, float(dist)))
            gx, gy = self.polar_to_xy_cm(angle, dist)

            # 未夾限本地 + 翻轉（注意順序）
            x_u = gx
            y_u = gy - self.PAD_Y0_CM
            x_u = -x_u
            y_u = self.ACT_H_CM - y_u

            # 不在範圍 → 不畫
            if not self._inside_pad(x_u, y_u):
                return

            # 顯示前最後一步才夾邊界
            x_local = max(-self.ACT_W_CM / 2, min(self.ACT_W_CM / 2, x_u))
            y_local = max(0.0, min(self.ACT_H_CM, y_u))

            # EMA & 繪圖
            xy = np.array([x_local, y_local], dtype=np.float32)
            ema = getattr(self, ema_xy_attr)
            ema = xy if ema is None else (self.ema_alpha_xy * xy + (1 - self.ema_alpha_xy) * ema)
            setattr(self, ema_xy_attr, ema)
            px, py = local_cm_to_px(float(ema[0]), float(ema[1]))

            p.setBrush(QtGui.QBrush(color))
            p.setPen(QtGui.QPen(color, 1))
            p.drawEllipse(QtCore.QPointF(px, py), self.point_size, self.point_size)

        # 模型（紅）
        draw_point(self.pred_angle, self.pred_dist,
                   QtGui.QColor(255, 80, 80), 'pred_xy_ema')

        # 如果處在「保留上一點」模式 → 畫凍結紅點
        if self.pred_hold_active and self.pred_hold_local is not None and self.pred_angle is None:
            hx, hy = self.pred_hold_local
            if self._inside_pad(hx, hy):
                px, py = local_cm_to_px(hx, hy)
                p.setBrush(QtGui.QBrush(QtGui.QColor(255, 80, 80)))
                p.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80), 1))
                p.drawEllipse(QtCore.QPointF(px, py), self.point_size, self.point_size)

        p.end()
        self.sector_label.setPixmap(pix)


# === Entry Point ===
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    kgl.setLib()

    def connect():
        try:
            dev = kgl.ksoclib.connectDevice()
            if dev == 'Unknow':
                raise RuntimeError
        except:
            rep = QtWidgets.QMessageBox.warning(
                None, 'Connect Failed', 'Retry?',
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            )
            if rep == QtWidgets.QMessageBox.Ok:
                connect()
    connect()

    ksp = SettingProc()
    cfg = SettingConfigs()
    cfg.Chip_ID   = kgl.ksoclib.getChipID().split(' ')[0]
    cfg.Processes = [
        'Reset Device', 'Gen Process Script', 'Gen Param Dict', 'Get Gesture Dict',
        'Set Script', 'Run SIC', 'Phase Calibration', 'Modulation On'
    ]
    cfg.setScriptDir(r'K60168-Test-00256-008-v0.0.8-20230717_60cm')
    ksp.startUp(cfg)

    vm  = ShowDataViewModel()
    win = ShowDataView(vm)
    recv = MultiResult4168BReceiver()
    recv.actions         = 1
    recv.rbank_ch_enable = 7
    recv.read_interrupt  = 0
    recv.clear_interrupt = 0
    FRM.setReceiver(recv)
    FRM.setUpdater(vm)
    FRM.trigger()
    FRM.start()

    win.resize(820, 900)
    win.show()
    sys.exit(app.exec_())
