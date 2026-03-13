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

# === Overall normalization parameters for distance-only model ===
mean_val = 9.702772
std_val  = 32.354683

# === ZeroMQ Client Setup ===
ctx  = zmq.Context()
sock = ctx.socket(zmq.REQ)
sock.connect("tcp://localhost:5555")

def ask_inference(seq: np.ndarray):
    """Send mode='dist' and normalized sequence, return distance."""
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
        self.dict     = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        self.detector = cv2.aruco.ArucoDetector(self.dict, cv2.aruco.DetectorParameters())

        # Sequence buffer
        self.seq_len = 20
        self.buffer  = deque(maxlen=self.seq_len)

        # Predictions
        self.pred_dist  = None
        self.april_dist = None

        # UI elements
        font = QtGui.QFont(); font.setPointSize(14); font.setBold(True)
        self.pred_label  = QtWidgets.QLabel('Predicted: -- cm')
        self.pred_label.setFont(font)
        self.april_label = QtWidgets.QLabel('AprilTag: -- cm')
        self.april_label.setFont(font)
        self.graph_label = QtWidgets.QLabel()
        self.graph_label.setFixedSize(400, 600)  # vertical graph
        self.legend_label = QtWidgets.QLabel('🔴 Model    🟢 AprilTag')
        self.legend_label.setFont(QtGui.QFont('', 12))
        self.cam_label    = QtWidgets.QLabel()
        self.cam_label.setFixedSize(800, 450)

        # Layout
        layout = QtWidgets.QVBoxLayout(self)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.pred_label); hl.addStretch(); hl.addWidget(self.april_label)
        layout.addLayout(hl)
        layout.addWidget(self.legend_label, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.graph_label, alignment=QtCore.Qt.AlignHCenter)
        layout.addWidget(self.cam_label,  alignment=QtCore.Qt.AlignHCenter)

        # Drawing params
        self.max_dist = 60.0

        # Start timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(int(1000/30))

    def updateRadar(self, res: Results):
        # get feature_map or raw_data
        arr = np.array(
            res['feature_map'].data if hasattr(res, 'feature_map') else res['raw_data'].data,
            dtype=np.float32
        )
        # collapse to single channel: if 2D, take as is; if 3D, take channel 0
        if arr.ndim == 2:
            proc = arr
        else:
            proc = arr[0]
        # push to buffer
        self.buffer.append(proc)
        if len(self.buffer) < self.seq_len:
            return

        seq = np.stack(self.buffer, axis=0).astype(np.float32)  # (T, H, W)
        seq = seq[:, None, :, :]  # (T, 1, H, W)
        # normalize
        seq_norm = (seq - mean_val) / std_val

        dist = ask_inference(seq_norm)
        self.pred_dist = dist
        self.pred_label.setText(f'Predicted: {dist:.1f} cm')
        self.updateGraph()

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None and ids.size > 0:
            tvecs = {}
            rvec0 = None

            # Collect tvec for tag0, tag1, tag3 and remember tag0's rvec
            for c, tid in zip(corners, ids.flatten()):
                if tid in (0, 1, 3):
                    rvecs_arr, tarr, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [c], self.markerLength, self.K, self.dist
                    )
                    tvecs[tid] = tarr[0, 0]
                    if tid == 0:
                        rvec0 = rvecs_arr[0, 0]

            # Only compute when tag0, tag1, tag3 are all seen
            if rvec0 is not None and {0, 1, 3}.issubset(tvecs):
                # 1) Compute tag plane normal (in camera coords)
                R_tag2cam, _ = cv2.Rodrigues(rvec0)
                tag_normal = R_tag2cam[:, 2]

                # 2) Project vector from tag0 to tag3 onto that plane
                vec03 = tvecs[3] - tvecs[0]
                proj = vec03 - (vec03.dot(tag_normal)) * tag_normal

                # 3) Distance = length of that projection
                dist_m = np.linalg.norm(proj)
                self.april_dist = round(dist_m * 100, 1)  # in cm

                # 4) Update label
                self.april_label.setText(f'AprilTag: {self.april_dist:.1f} cm')

                # 5) Redraw graph
                self.updateGraph()

        # draw camera image (unchanged)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            rgb.data, rgb.shape[1], rgb.shape[0],
            rgb.shape[1]*3, QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            self.cam_label.size(), QtCore.Qt.KeepAspectRatio
        )
        self.cam_label.setPixmap(pix)

    def updateGraph(self):
        w, h = self.graph_label.width(), self.graph_label.height()
        pix = QtGui.QPixmap(w, h)
        pix.fill(QtCore.Qt.black)
        p = QtGui.QPainter(pix)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # draw two vertical axes
        x1 = w * 1/3
        x2 = w * 2/3
        bottom = h - 20
        top    = 20

        pen = QtGui.QPen(QtCore.Qt.white, 2)
        p.setPen(pen)
        p.drawLine(int(x1), int(bottom), int(x1), int(top))
        p.drawLine(int(x2), int(bottom), int(x2), int(top))

        # draw predicted distance (red)
        if self.pred_dist is not None:
            y = bottom - ((self.pred_dist / self.max_dist) * (bottom - top))
            brush = QtGui.QBrush(QtCore.Qt.red)
            p.setBrush(brush)
            p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(QtCore.QPointF(x1, y), 10, 10)

        # draw AprilTag distance (green)
        if self.april_dist is not None:
            y = bottom - ((self.april_dist / self.max_dist) * (bottom - top))
            brush = QtGui.QBrush(QtCore.Qt.green)
            p.setBrush(brush)
            p.setPen(QtCore.Qt.NoPen)
            p.drawEllipse(QtCore.QPointF(x2, y), 10, 10)

        p.end()
        self.graph_label.setPixmap(pix)

# === Entry Point ===

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    kgl.setLib()

    # connect device
    def connect():
        try:
            dev = kgl.ksoclib.connectDevice()
            if dev == 'Unknow':
                raise RuntimeError
        except:
            rep = QtWidgets.QMessageBox.warning(
                None, 'Connect Failed', 'Retry?',
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            if rep == QtWidgets.QMessageBox.Ok:
                connect()
    connect()

    # setup radar
    ksp = SettingProc()
    cfg = SettingConfigs()
    cfg.Chip_ID   = kgl.ksoclib.getChipID().split(' ')[0]
    cfg.Processes = [
        'Reset Device', 'Gen Process Script', 'Gen Param Dict', 'Get Gesture Dict',
        'Set Script', 'Run SIC', 'Phase Calibration', 'Modulation On'
    ]
    cfg.setScriptDir(r'K60168-Test-00256-008-v0.0.8-20230717_120cm')
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
