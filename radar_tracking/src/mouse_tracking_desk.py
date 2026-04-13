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

        # ========== BUFFER ==========
        self.seq_len = 20
        self.buffer = deque(maxlen=self.seq_len)

        # ========== Pred ==========
        self.pred_angle = None
        self.pred_dist  = None

        # ========== UI 元件 ==========
        main_layout = QtWidgets.QVBoxLayout(self)

        # 顯示模型紅點（不顯示 angle / dist / xy）
        self.sector_label = QtWidgets.QLabel()
        self.sector_label.setFixedSize(600, 600)
        main_layout.addWidget(self.sector_label, alignment=QtCore.Qt.AlignHCenter)

        # 模型紅點標示
        self.legend_label = QtWidgets.QLabel("🔴 Radar Prediction")
        self.legend_label.setFont(QtGui.QFont('', 12))
        main_layout.addWidget(self.legend_label, alignment=QtCore.Qt.AlignHCenter)

        # ========== 全域滑鼠模式(F8) ==========
        self.mouse_mode_enabled = False
        self.toggle_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence('F8'), self)
        self.toggle_shortcut.activated.connect(self.toggle_mouse_mode)

        self.mode_label = QtWidgets.QLabel("模式：視窗控制 (F8 切換)")
        self.mode_label.setFont(QtGui.QFont('', 11))
        main_layout.insertWidget(0, self.mode_label)

        # 更新滑鼠位置計時器
        self.cursor_timer = QtCore.QTimer(self)
        self.cursor_timer.setInterval(30)  # 33Hz smoother
        self.cursor_timer.timeout.connect(self._cursor_step)

        # ========== 16×16 方形空間 (中心為 0,0) ==========
        # 你的原本是 20×16，現在改成 16×16，座標限制：
        # x ∈ [-8, +8]
        # y ∈ [-8, +8]
        self.AR = 8.0     # half-range

        # === v3 使用的真實 Pad 幾何參數 ===
        self.ACT_W_CM = 20.0  # 實際操作區寬度
        self.ACT_H_CM = 16.0  # 實際操作區高度
        self.PAD_Y0_CM = 12.0  # Pad 起始位置 (v3 原版)

        self.BOX_W = 16.0
        self.BOX_H = 16.0

        # 小圓與大圓（單位: 格）
        self.big_r  = 6.0               # 大圓半徑 = 整個方形最大範圍
        self.small_r = self.big_r * 0.35  # 小圓半徑 (死區)

        # ========== 繪製與平滑 ==========
        self.pred_xy_ema = None
        self.ema_alpha_xy = 0.3

        # ========== Energy gating (保留你的後處理邏輯) ==========
        self.ema_energy = None
        self.ema_alpha_energy = 0.4

        # 原本的掉出範圍後 hold 點邏輯保留
        self.pred_hold_local = None
        self.pred_hold_active = False
        self.boundary_tol_cm = 1
        self.clear_outside_frames = 5
        self._outside_cnt = 0

        # ========== 滑鼠移動參數 ==========
        self.base_speed = 22.0  # max speed（可調）

        # 搖桿方向向量
        self.joy_dir = None  # ← 這行一定要有，否則游標永遠不動

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

    # === 滑鼠控制模式切換（F8） ===
    def toggle_mouse_mode(self):
        self.mouse_mode_enabled = not self.mouse_mode_enabled

        if self.mouse_mode_enabled:
            self.mode_label.setText("模式：滑鼠控制 (F8 切換)")
            self.cursor_timer.start()
        else:
            self.mode_label.setText("模式：視窗控制 (F8 切換)")
            self.cursor_timer.stop()

    # === 游標移動 (依大小圓 + 距離調速) ===
    def _cursor_step(self):
        if not self.mouse_mode_enabled:
            return
        if self.pred_hold_local is None:
            return

        x_local, y_local = self.pred_hold_local  # 這裡的座標已經是 [-8, 8] 范圍的搖桿座標

        # 距離中心 r（單位：格）
        r = math.sqrt(x_local * x_local + y_local * y_local)
        if r <= self.small_r:
            # 小圓內：不動
            return

        # 小圓外 → 根據 r 線性放大速度
        norm = (r - self.small_r) / (self.big_r - self.small_r)
        norm = max(0.0, min(1.0, norm))  # 0~1

        speed = self.base_speed * norm

        # 方向向量（以中心為原點）
        dx = x_local / r
        dy = y_local / r

        # Qt 螢幕座標：x 往右增加，y 往下增加
        cx, cy = QtGui.QCursor.pos().x(), QtGui.QCursor.pos().y()
        nx = cx + dx * speed
        ny = cy - dy * speed  # y_local > 0 代表往上，所以要用減號

        QtGui.QCursor.setPos(int(nx), int(ny))

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
        if len(self.buffer) < MAX_BUFFER:
            self.pred_hold_local = None
            self.pred_xy_ema = None
            self.updateXY()
            return

        # === Step 4: 模型推論 ===
        seq = np.stack(list(self.buffer), axis=0).astype(np.float32)
        seq_norm = (seq - mean_arr[None, :, None, None]) / std_arr[None, :, None, None]

        angle = ask_angle(seq_norm)
        dist = ask_dist(seq_norm[:, 0:1])

        gx, gy = self.polar_to_xy_cm(float(angle), float(dist))

        # === Step 5: Energy 門檻 (EMA) ===
        d = max(1.0, min(RMAX, float(dist)))
        LOW_ENERGY_TH = TH_NEAR - (d / RMAX) * (TH_NEAR - TH_FAR)

        if self.ema_energy is None:
            self.ema_energy = total_energy
        else:
            self.ema_energy = (1 - self.ema_alpha_energy) * self.ema_energy + \
                              self.ema_alpha_energy * total_energy

        detected = self.ema_energy >= LOW_ENERGY_TH

        if not detected:
            if self.pred_hold_local is not None:
                self.updateXY()
                return

            self.pred_hold_local = None
            self.pred_xy_ema = None
            self.updateXY()
            return

        # === Step 7: 使用 v3 的 Pad 計算 (真實世界 20×16 cm) ===

        PAD_Y0 = self.PAD_Y0_CM       # Pad 下緣
        PAD_H  = self.ACT_H_CM        # Pad 高度 16 cm
        PAD_W  = self.ACT_W_CM        # Pad 寬度 20 cm

        # --- v3 基礎座標翻轉 ---
        x_u = -gx
        y_u = PAD_H - (gy - PAD_Y0)

        # --- 夾到 Pad cm 邊界 ---
        x_local_cm = max(-PAD_W/2, min(PAD_W/2, x_u))
        y_local_cm = max(0.0,       min(PAD_H,     y_u))

        # ==========================================================
        # ★ Step 7b：圓形限制是在 "Pad cm 座標空間" 內切割圓形 ★
        # ==========================================================

        # Pad 中心點 (cm)
        cx = 0.0
        cy = PAD_H / 2.0

        # 從 Pad 中心偏移的向量
        dx = x_local_cm - cx
        dy = y_local_cm - cy

        # 實際半徑 r（仍在 cm 單位）
        r = math.sqrt(dx*dx + dy*dy)

        # 最大允許半徑（cm）
        R_cm = self.big_r      # big_r 應視為 "cm 半徑"，而不是格數

        # 若超過圓形 → 夾在圓形邊界
        if r > R_cm:
            scale = R_cm / r
            dx *= scale
            dy *= scale
            x_local_cm = cx + dx
            y_local_cm = cy + dy

        # ==========================================================
        # ★ Step 7c：將 cm → GUI joystick 格子（顯示用）
        # ==========================================================

        # 最後轉換成你原本的 [-8,+8] 顯示座標
        x_local = (x_local_cm / (PAD_W/2)) * self.AR
        y_local = ((y_local_cm - PAD_H/2) / (PAD_H/2)) * self.AR
        # y_local = -y_local   # 與 GUI 規則一致

        # 存起來（給 updateXY 畫點）
        self.pred_hold_local = (x_local, y_local)
        self.joy_dir = (dx/(r+1e-6), dy/(r+1e-6))

        self.updateXY()



    # ===== 核心：把扇形改成方形 XY 顯示 =====
    def updateXY(self):
        w, h = self.sector_label.width(), self.sector_label.height()
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.black)

        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # ========== 基本方形參數 ==========
        BOX = self.AR * 2  # 16 格
        margin = 20  # 邊界
        s = (w - 2 * margin) / BOX  # 每格像素大小

        # 方形左上角座標
        left = margin
        top = margin
        right = left + BOX * s
        bottom = top + BOX * s

        # ========== 畫外框 ==========
        p.setPen(QtGui.QPen(QtCore.Qt.white, 3))
        p.drawRect(left, top, BOX * s, BOX * s)

        # ========== 畫格線 ==========
        p.setPen(QtGui.QPen(QtCore.Qt.gray, 1))
        for i in range(1, int(BOX)):
            x = left + i * s
            p.drawLine(x, top, x, bottom)
            y = top + i * s
            p.drawLine(left, y, right, y)

        # ========== 大圓 / 小圓 ==========
        center_x = left + BOX * s / 2
        center_y = top + BOX * s / 2

        big_r_px = self.big_r * s
        small_r_px = self.small_r * s

        # 大圓
        p.setPen(QtGui.QPen(QtCore.Qt.white, 2))
        p.drawEllipse(QtCore.QPointF(center_x, center_y), big_r_px, big_r_px)

        # 小圓（死區）
        p.setPen(QtGui.QPen(QtCore.Qt.darkGray, 2))
        p.drawEllipse(QtCore.QPointF(center_x, center_y), small_r_px, small_r_px)

        # ========== 畫紅點 (從 pred_hold_local or pred_xy_ema) ==========
        if self.pred_hold_local is not None:
            x_local, y_local = self.pred_hold_local

            # EMA 平滑
            xy = np.array([x_local, y_local], dtype=np.float32)
            ema = self.pred_xy_ema
            ema = xy if ema is None else (self.ema_alpha_xy * xy + (1 - self.ema_alpha_xy) * ema)
            self.pred_xy_ema = ema
            x_local, y_local = float(ema[0]), float(ema[1])

            # 映射到 px
            px = center_x + x_local * s
            py = center_y - y_local * s  # y 軸為反向

            p.setBrush(QtGui.QBrush(QtGui.QColor(255, 80, 80)))
            p.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80)))
            p.drawEllipse(QtCore.QPointF(px, py), 40, 40)

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
