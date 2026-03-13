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

# === Overall normalization parameters for angle-only model ===
mean_val = 27.205647
std_val  = 74.960556

# === ZeroMQ Client Setup ===
ctx  = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")

def ask_inference(seq: np.ndarray):
    """Send mode='angle' and normalized sequence, return angle."""
    sock.send(pickle.dumps(('angle', seq), protocol=4))
    angle, _ = sock.recv_pyobj()
    return angle

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
        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        self.detector = cv2.aruco.ArucoDetector(self.dict, cv2.aruco.DetectorParameters())

        # Sequence buffer
        self.seq_len = 20
        self.buffer  = deque(maxlen=self.seq_len)

        # Predictions
        self.pred_angle  = None
        self.april_angle = None

        # UI elements
        font = QtGui.QFont(); font.setPointSize(14); font.setBold(True)
        self.pred_label  = QtWidgets.QLabel('Predicted: --°')
        self.pred_label.setFont(font)
        self.april_label = QtWidgets.QLabel('AprilTag: --°')
        self.april_label.setFont(font)
        self.sector_label = QtWidgets.QLabel()
        self.sector_label.setFixedSize(800, 400)
        self.legend_label = QtWidgets.QLabel('🔴 Model Prediction    🟢 AprilTag')
        self.legend_label.setFont(QtGui.QFont('', 12))
        self.cam_label    = QtWidgets.QLabel()
        self.cam_label.setFixedSize(800, 450)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.pred_label); hl.addStretch(); hl.addWidget(self.april_label)
        layout.addLayout(hl)
        layout.addWidget(self.sector_label, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.legend_label, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.cam_label, alignment=QtCore.Qt.AlignHCenter)

        # Drawing params
        self.max_dist     = 30.0
        self.sector_ratio = 1.5

        # Start timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(int(1000/30))

    def updateRadar(self, res: Results):
        # get feature_map or raw_data
        arr = np.array(
            res['feature_map'].data
            if hasattr(res, 'feature_map')
            else res['raw_data'].data,
            dtype=np.float32
        )
        proc = np.log10(arr + 1e-6) if arr.ndim == 2 else arr
        if proc.ndim == 2:
            proc = np.stack([proc, proc], 0)

        # push to buffer
        self.buffer.append(proc)
        if len(self.buffer) < self.seq_len:
            return

        seq = np.stack(self.buffer, axis=0).astype(np.float32)  # shape (T,2,H,W)
        # normalize only angle channel: here we normalize entire array
        seq_norm = (seq - mean_val) / std_val

        # inference for angle only
        angle = ask_inference(seq_norm)
        self.pred_angle = angle
        self.pred_label.setText(f'Predicted: {angle:.1f}°')
        self.updateSector()

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and ids.size > 0:
            # 收集 tag0、tag1、tag3 的 tvec，以及 tag0 的 rvec
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

            # 僅在 tag0、tag1、tag3 全部偵測到時才計算
            if rvec0 is not None and {0, 1, 3}.issubset(tvecs):
                # 1) 計算 Tag 平面的法向量 (在 camera 座標)
                R_tag2cam, _ = cv2.Rodrigues(rvec0)
                tag_normal = R_tag2cam[:, 2]

                # 2) 以 Tag0→Tag1 在地板平面的投影作為「右向」單位向量
                base = tvecs[1] - tvecs[0]
                base_proj = base - (base.dot(tag_normal)) * tag_normal
                base_norm = base_proj / np.linalg.norm(base_proj)

                # 3) 前向 = 地板法向量 × 右向
                forward = np.cross(tag_normal, base_norm)
                forward /= np.linalg.norm(forward)

                # 4) 計算 Tag3→Tag0 向量投影到地板平面
                vec03 = tvecs[3] - tvecs[0]
                proj = vec03 - (vec03.dot(tag_normal)) * tag_normal

                # 5) 計算水平夾角：atan2(投影向量在「右向」的分量, 在「前向」的分量)
                x = proj.dot(base_norm)
                y = proj.dot(forward)
                angle_rad = math.atan2(x, y)
                self.april_angle = round(math.degrees(angle_rad), 1)

                # 更新顯示
                self.april_label.setText(f'AprilTag: {self.april_angle:.1f}°')
                self.updateSector()

        # 顯示 camera 畫面（原有程式不變）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            rgb.data, rgb.shape[1], rgb.shape[0],
            rgb.shape[1]*3, QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.cam_label.size(), QtCore.Qt.KeepAspectRatio
        )
        self.cam_label.setPixmap(pix)

    def updateSector(self):
        w, h = self.sector_label.width(), self.sector_label.height()
        pix = QtGui.QPixmap(w, h); pix.fill(QtCore.Qt.black)
        p = QtGui.QPainter(pix); p.setRenderHint(QtGui.QPainter.Antialiasing)
        cpt = QtCore.QPointF(w/2, h)
        r   = min(w, h) * self.sector_ratio / 2

        # draw arc
        p.setPen(QtGui.QPen(QtCore.Qt.white, 3))
        p.drawArc(int(cpt.x()-r), int(cpt.y()-r), int(2*r), int(2*r), (90-45)*16, 90*16)

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

        # draw model prediction as a line (red)
        if self.pred_angle is not None:
            rad = math.radians(self.pred_angle)
            xe = cpt.x() + r*math.sin(rad)
            ye = cpt.y() - r*math.cos(rad)
            p.setPen(QtGui.QPen(QtCore.Qt.red, 4))
            p.drawLine(cpt, QtCore.QPointF(xe, ye))

        # draw AprilTag detection as a line (green)
        if self.april_angle is not None:
            rad = math.radians(self.april_angle)
            xe = cpt.x() + r*math.sin(rad)
            ye = cpt.y() - r*math.cos(rad)
            p.setPen(QtGui.QPen(QtCore.Qt.green, 4))
            p.drawLine(cpt, QtCore.QPointF(xe, ye))

        p.end()
        self.sector_label.setPixmap(pix)

# === Entry Point ===

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    kgl.setLib()

    def connect():
        try:
            dev = kgl.ksoclib.connectDevice()
            if dev == 'Unknow': raise RuntimeError
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
