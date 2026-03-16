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
        self.PAD_Y0_CM = 12.0  # 距雷達最近 12cm 起算（矩形底邊）
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

        # 畫點大小與軌跡/平滑
        self.point_size = 20
        self.ema_alpha_xy = 0.2
        self.pred_xy_ema = None

        # === Grid 與「掉門檻時保留上一點」 ===
        self.grid_cm = 2.0  # 每格 2 cm（正方形）
        self.pred_hold_local = None  # (x_local, y_local) 以 cm 為單位
        self.pred_hold_active = False  # True 代表用「保留點」顯示紅點

        self.boundary_tol_cm = 1  # 邊界緩衝帶（寬鬆範圍的外擴量）
        self.clear_outside_frames = 5  # 需連續在寬鬆範圍外多少幀才清掉
        self._outside_cnt = 0  # 計數：連續在寬鬆範圍外的幀數

        # === 新增平滑機制 ===
        self.ema_energy = None  # 用來平滑能量，避免閃爍
        self.ema_alpha_energy = 0.4  # 能量EMA係數（越小越穩定）

    def _inside_pad(self, x_local: float, y_local: float) -> bool:
        return (-self.ACT_W_CM / 2 <= x_local <= self.ACT_W_CM / 2) and (0.0 <= y_local <= self.ACT_H_CM)

    def _inside_pad_strict(self, x_local: float, y_local: float) -> bool:
        # 嚴格：用來「接受新點」的判斷
        return (-self.ACT_W_CM / 2 <= x_local <= self.ACT_W_CM / 2) and (0.0 <= y_local <= self.ACT_H_CM)

    def _inside_pad_loose(self, x_local: float, y_local: float) -> bool:
        # 寬鬆：用來「保留/清除」的判斷（四邊各加 boundary_tol_cm 緩衝）
        t = self.boundary_tol_cm
        return (-self.ACT_W_CM / 2 - t <= x_local <= self.ACT_W_CM / 2 + t) and (-t <= y_local <= self.ACT_H_CM + t)

    # ---- Helpers: polar -> xy (cm) -> canvas px ----
    def polar_to_xy_cm(self, angle_deg: float, dist_cm: float):
        rad = math.radians(angle_deg)
        x = dist_cm * math.sin(rad)  # 右正
        y = dist_cm * math.cos(rad)  # 前正
        return x, y

    def updateRadar(self, res: Results):
        # === Step 1: 取得雷達原始資料 ===
        arr = np.array(
            res['feature_map'].data
            if hasattr(res, 'feature_map')
            else res['raw_data'].data,
            dtype=np.float32
        )
        total_energy = float(np.sum(np.abs(arr)))
        proc = np.log10(arr + 1e-6) if arr.ndim == 2 else arr
        if proc.ndim == 2:
            proc = np.stack([proc, proc], 0)

        # === Step 2: 設定參數 ===
        MAX_BUFFER = 20
        TH_NEAR = 40000.0
        TH_FAR = 10000.0
        RMAX = 30.0  # cm

        # === Step 3: 更新 buffer ===
        self.buffer.append(proc)
        if len(self.buffer) > MAX_BUFFER:
            self.buffer.popleft()
        if len(self.buffer) < MAX_BUFFER:
            self.pred_angle = None
            self.pred_dist = None
            self.pred_xy_ema = None
            self.updateXY()
            return

        # === Step 4: 模型推論 ===
        seq = np.stack(list(self.buffer), axis=0).astype(np.float32)
        seq_norm = (seq - mean_arr[None, :, None, None]) / std_arr[None, :, None, None]
        angle = ask_angle(seq_norm)
        seq_dist = seq_norm[:, 0:1, :, :]
        dist = ask_dist(seq_dist)

        gx, gy = self.polar_to_xy_cm(float(angle), float(dist))
        x_local_raw = gx
        y_local_raw = gy - self.PAD_Y0_CM

        # === Step 5: 動態能量門檻 + 平滑能量 ===
        d = max(1.0, min(RMAX, float(dist)))
        LOW_ENERGY_TH = TH_NEAR - (d / RMAX) * (TH_NEAR - TH_FAR)

        # 用 EMA 平滑能量，避免閃爍
        if self.ema_energy is None:
            self.ema_energy = total_energy
        else:
            self.ema_energy = (1 - self.ema_alpha_energy) * self.ema_energy + self.ema_alpha_energy * total_energy

        # === Step 6: 使用平滑後的能量判斷是否通過門檻 ===
        detected = self.ema_energy >= LOW_ENERGY_TH
        print(f"raw_energy={total_energy:.1f}, ema_energy={self.ema_energy:.1f}, th={LOW_ENERGY_TH:.1f}")

        # === Step 7: 若未偵測，保留上一點 ===
        if not detected:
            if self.pred_hold_local is not None:
                hx, hy = self.pred_hold_local
                if self._inside_pad_loose(hx, hy):
                    self._outside_cnt = 0
                    self.pred_hold_active = True
                    self.pred_angle = None
                    self.pred_dist = None
                    self.pred_label.setText('Predicted: --°, -- cm')
                    self.pred_xy_label.setText(
                        f'Pred Grid XY: {hx / self.grid_cm:.1f}, {hy / self.grid_cm:.1f} (cells)')
                    self.updateXY()
                    return
                else:
                    # 在寬鬆範圍外 → 累積離開幀數
                    self._outside_cnt += 1
                    if self._outside_cnt < self.clear_outside_frames:
                        self.pred_angle = None
                        self.pred_dist = None
                        self.pred_label.setText('Predicted: --°, -- cm')
                        self.pred_xy_label.setText(
                            f'Pred Grid XY: {hx / self.grid_cm:.1f}, {hy / self.grid_cm:.1f} (cells)')
                        self.updateXY()
                        return
            # 超過外部計數 → 清除紅點
            self.pred_hold_local = None
            self.pred_hold_active = False
            self.pred_xy_ema = None
            self.pred_label.setText('Predicted: --°, -- cm')
            self.pred_xy_label.setText('Pred Grid XY: --, -- (cells)')
            self.updateXY()
            return

        # === Step 8: 僅在矩形有效範圍內顯示 ===
        x_u = -x_local_raw
        y_u = self.ACT_H_CM - y_local_raw

        if not self._inside_pad(x_u, y_u):
            # 超出有效顯示區域 → 立即清除紅點
            self.pred_hold_local = None
            self.pred_hold_active = False
            self.pred_xy_ema = None
            self.pred_angle = None
            self.pred_dist = None
            self.pred_label.setText('Predicted: --°, -- cm')
            self.pred_xy_label.setText('Pred Grid XY: --, -- (cells)')
            self.updateXY()
            return

        # === Step 9: 正常更新顯示 ===
        self.pred_hold_active = False
        self.pred_angle, self.pred_dist = angle, dist
        self.pred_label.setText(f'Predicted: {angle:.1f}°, {dist:.1f} cm')

        x_local = max(-self.ACT_W_CM / 2, min(self.ACT_W_CM / 2, x_u))
        y_local = max(0.0, min(self.ACT_H_CM, y_u))
        self.pred_hold_local = (x_local, y_local)
        self.pred_xy_label.setText(
            f'Pred Grid XY: {x_local / self.grid_cm:.1f}, {y_local / self.grid_cm:.1f} (cells)')
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

            # ① 寬鬆判斷：完全在寬鬆範圍外 → 不畫
            if not self._inside_pad_loose(x_u, y_u):
                return

            # ② 嚴格/寬鬆：嚴格外但寬鬆內 → 吸附到邊界再畫；嚴格內 → 直接用
            if not self._inside_pad_strict(x_u, y_u):
                # 吸附到邊界（避免一碰邊就消失）
                x_local = max(-self.ACT_W_CM / 2, min(self.ACT_W_CM / 2, x_u))
                y_local = max(0.0, min(self.ACT_H_CM, y_u))
            else:
                x_local, y_local = x_u, y_u

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
