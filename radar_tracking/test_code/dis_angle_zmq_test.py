import sys
import math
import pickle
import zmq
import numpy as np
import cv2
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
    """Send angle request, return angle."""
    sock.send(pickle.dumps(('angle', seq), protocol=4))
    angle, _ = sock.recv_pyobj()
    return angle

def ask_dist(seq: np.ndarray):
    """Send distance request, return dist."""
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

        # Camera & AprilTag
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        calib = np.load('../calib.npz')
        self.K, self.dist = calib['K'], calib['dist']
        self.markerLength = 0.03
        self.dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_16H5
        )
        self.detector = cv2.aruco.ArucoDetector(
            self.dict, cv2.aruco.DetectorParameters()
        )

        # Sequence buffer
        self.seq_len = 20
        self.buffer  = deque(maxlen=self.seq_len)

        # Predictions
        self.pred_angle  = None
        self.pred_dist   = None
        self.april_angle = None
        self.april_dist  = None

        # UI elements
        font = QtGui.QFont(); font.setPointSize(14); font.setBold(True)
        self.pred_label  = QtWidgets.QLabel('Predicted: --°, -- cm')
        self.pred_label.setFont(font)
        self.april_label = QtWidgets.QLabel('AprilTag: --°, -- cm')
        self.april_label.setFont(font)
        self.sector_label = QtWidgets.QLabel()
        self.sector_label.setFixedSize(800, 400)
        self.legend_label = QtWidgets.QLabel(
            '🔴 Model Prediction    🟢 AprilTag Detection'
        )
        self.legend_label.setFont(QtGui.QFont('', 12))
        self.cam_label    = QtWidgets.QLabel()
        self.cam_label.setFixedSize(800, 450)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.pred_label)
        hl.addStretch()
        hl.addWidget(self.april_label)
        layout.addLayout(hl)
        layout.addWidget(self.sector_label, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.legend_label, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.cam_label, alignment=QtCore.Qt.AlignHCenter)

        # Drawing params
        self.max_dist     = 30.0
        self.sector_ratio = 1.5
        self.point_size   = 15

        # Start timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(int(1000/30))

    def updateRadar(self, res: Results):
        # get radar data
        arr = np.array(
            res['feature_map'].data
            if hasattr(res, 'feature_map')
            else res['raw_data'].data,
            dtype=np.float32
        )
        proc = np.log10(arr + 1e-6) if arr.ndim == 2 else arr
        if proc.ndim == 2:
            proc = np.stack([proc, proc], 0)  # make 2-channel

        # push to buffer
        self.buffer.append(proc)
        if len(self.buffer) < self.seq_len:
            return

        # build sliding window sequence
        seq = np.stack(self.buffer, axis=0).astype(np.float32)  # (T,2,H,W)

        # normalize per channel
        seq_norm = (
            seq - mean_arr[None, :, None, None]
        ) / std_arr[None, :, None, None]

        # ask two models
        angle = ask_angle(seq_norm)
        # for distance, take only channel 0
        seq_dist = seq_norm[:, 0:1, :, :]  # (T,1,H,W)
        dist  = ask_dist(seq_dist)

        # update labels & state
        self.pred_angle, self.pred_dist = angle, dist
        self.pred_label.setText(
            f'Predicted: {angle:.1f}°, {dist:.1f} cm'
        )
        self.updateSector()

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and ids.size > 0:
            # 1) 收集 tag0、tag1、tag3 的 tvec，並存下 tag0 的 rvec
            tvecs = {}
            rvec0 = None
            for c, tid in zip(corners, ids.flatten()):
                if tid in (0, 1, 3):
                    rvecs_arr, tarr, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [c], self.markerLength, self.K, self.dist
                    )
                    tvecs[tid] = tarr[0, 0]
                    if tid == 0:
                        rvec0 = rvecs_arr[0, 0]

            # 2) 只有在 tag0、tag1、tag3 都偵測到時才計算
            if rvec0 is not None and {0, 1, 3}.issubset(tvecs):
                # 推出 Tag 平面的法向量（在 camera 座標下）
                R_tag2cam, _ = cv2.Rodrigues(rvec0)  # tag→camera
                tag_normal = R_tag2cam[:, 2]         # 第三欄 = tag z 軸（垂直向上）

                # 以 Tag0→Tag1 在地板平面上的投影作為「右向」單位向量
                base = tvecs[1] - tvecs[0]
                base_proj = base - (base.dot(tag_normal)) * tag_normal
                base_norm = base_proj / np.linalg.norm(base_proj)

                # 前向單位向量 = 地板法向量 × 右向
                forward = np.cross(tag_normal, base_norm)
                forward /= np.linalg.norm(forward)

                # 計算 Tag3→Tag0 向量投影到地板平面
                vec03 = tvecs[3] - tvecs[0]
                proj = vec03 - (vec03.dot(tag_normal)) * tag_normal

                # 水平距離 (cm)
                dist_m = np.linalg.norm(proj)
                self.april_dist = round(dist_m * 100, 1)

                # 水平角度 (deg)：atan2(右向分量, 前向分量)
                x = proj.dot(base_norm)
                y = proj.dot(forward)
                angle_rad = math.atan2(x, y)
                self.april_angle = round(math.degrees(angle_rad), 1)

                self.april_label.setText(
                    f'AprilTag: {self.april_angle:.1f}°, {self.april_dist:.1f} cm'
                )

        # 以下為原有的 camera 顯示，不變
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            rgb.data, rgb.shape[1], rgb.shape[0],
            rgb.strides[0], QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.cam_label.size(), QtCore.Qt.KeepAspectRatio
        )
        self.cam_label.setPixmap(pix)

    def updateSector(self):
        w, h = self.sector_label.width(), self.sector_label.height()
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.black)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        cpt = QtCore.QPointF(w/2, h)
        r   = min(w, h) * self.sector_ratio / 2

        # draw arc
        p.setPen(QtGui.QPen(QtCore.Qt.white, 3))
        p.drawArc(
            int(cpt.x()-r), int(cpt.y()-r),
            int(2*r), int(2*r),
            (90-45)*16, 90*16
        )

        # draw spokes & labels
        p.setPen(QtGui.QPen(QtCore.Qt.gray, 2, QtCore.Qt.DashLine))
        for ang in range(-45, 46, 15):
            rad = math.radians(ang)
            xe = cpt.x() + r*math.sin(rad)
            ye = cpt.y() - r*math.cos(rad)
            p.drawLine(cpt, QtCore.QPointF(xe, ye))
            xt = cpt.x() + (r+20)*math.sin(rad)
            yt = cpt.y() - (r+20)*math.cos(rad)
            p.setPen(QtGui.QPen(QtCore.Qt.white))
            p.drawText(QtCore.QPointF(xt-10, yt), f"{ang}°")

        # draw model prediction (red)
        if self.pred_angle is not None and self.pred_dist is not None:
            rad = math.radians(self.pred_angle)
            rr  = min(self.pred_dist/self.max_dist, 1.0) * r
            px  = cpt.x() + rr*math.sin(rad)
            py  = cpt.y() - rr*math.cos(rad)
            p.setBrush(QtGui.QBrush(QtCore.Qt.red))
            p.drawEllipse(QtCore.QPointF(px, py), self.point_size, self.point_size)

        # draw AprilTag detection (green)
        if self.april_angle is not None and self.april_dist is not None:
            rad = math.radians(self.april_angle)
            rr  = min(self.april_dist/self.max_dist, 1.0) * r
            px  = cpt.x() + rr*math.sin(rad)
            py  = cpt.y() - rr*math.cos(rad)
            p.setBrush(QtGui.QBrush(QtCore.Qt.green))
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
