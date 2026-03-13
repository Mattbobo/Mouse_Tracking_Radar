from KKT_Module import kgl  # Core radar control library
from KKT_Module.DataReceive.Core import Results  # Radar data structure
from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver  # Data receiver for 1Tx-2Rx
from KKT_Module.FiniteReceiverMachine import FRM  # Finite receiver state machine
from KKT_Module.SettingProcess.SettingConfig import SettingConfigs  # Script configuration template
from KKT_Module.SettingProcess.SettingProccess import SettingProc  # Script execution helper
from KKT_Module.GuiUpdater.GuiUpdater import Updater  # Base class for radar plot updater
from KKT_UI.KKTGraph import ShowADCRaw, ShowFeatureMap  # Radar plot widgets
from KKT_UI.QTWidget.MainWindows import KKTMainWindow  # Base Qt main window
from PySide2 import QtWidgets, QtCore, QtGui  # Qt GUI modules
import cv2  # OpenCV for camera and AprilTag detection
import numpy as np  # Numerical computations
import h5py  # HDF5 storage
import time  # Timing functions
import os  # Filesystem operations
import csv  # CSV file handling

# === mmWave 連線與錯誤重試 ===
def connect(): #相機高度線
    try:
        device = kgl.ksoclib.connectDevice()
        if device == 'Unknow':
            if QtWidgets.QMessageBox.warning(
                    None, 'Unknown Device',
                    'Please reconnect device and try again',
                    QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
                ) == QtWidgets.QMessageBox.Ok:
                connect()
    except:
        if QtWidgets.QMessageBox.warning(
                None, 'Connection Failed',
                'Please reconnect device and try again',
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            ) == QtWidgets.QMessageBox.Ok:
            connect()

# === 通用屬性設定 ===
def setProperty(obj: object, **kwargs):
    for k, v in kwargs.items():
        if hasattr(obj, k): setattr(obj, k, v)

# === 設置雷達腳本 ===
def setScript(setting_name: str):
    ksp = SettingProc()
    cfg = SettingConfigs()
    cfg.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
    cfg.Processes = [
        'Reset Device', 'Gen Process Script', 'Gen Param Dict', 'Get Gesture Dict',
        'Set Script', 'Run SIC', 'Phase Calibration', 'Modulation On'
    ]
    cfg.setScriptDir(setting_name)
    ksp.startUp(cfg)

# === 雷達數據信號 ===
class ShowDataSentinel(QtCore.QObject):
    sig_update_plot = QtCore.Signal(object)

class ShowDataViewModel(Updater):
    def __init__(self):
        super().__init__()
        self.sentinel = ShowDataSentinel()
    def update(self, res: Results):
        self.sentinel.sig_update_plot.emit(res)

class ShowDataView(KKTMainWindow):
    def __init__(self, vm, show_data_type='feature_map'):
        super().__init__(title='Radar & Camera Recorder (Angle & Distance)')
        calib = np.load('calib.npz')
        self.K = calib['K']; self.dist = calib['dist']
        self.fx = self.K[0,0]; self.cx = self.K[0,2]

        self.align_px_thresh = 5
        self.baseline_v_thresh = 5
        self.markerLength = 0.03  # (m)

        # 角度與距離記錄
        self.angles = []
        self.distances = []
        self.current_angle = np.nan
        self.current_distance = np.nan

        self.vm = vm
        self.vm.sentinel.sig_update_plot.connect(self.updatePlots)
        self.show_data_type = show_data_type

        if show_data_type == 'raw_data':
            kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
            self.data_widget = ShowADCRaw.MultiRawDataPLotsWidget()
        else:
            kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)
            self.data_widget = ShowFeatureMap.MultiFeatureMapPlotsWidget()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.desired_fps = 60.0
        self.frame_interval = 1.0 / self.desired_fps

        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)

        self.is_recording = False
        self.total_frames = 0
        self.recorded_count = 0
        self.last_radar = None
        self.video_writer = None
        self.h5file = None
        self.h5ds = None

        self.setup()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(int(self.frame_interval * 1000))

    def setup(self):
        w_view, h_view = 960, 540
        self.wg = QtWidgets.QWidget()
        self.setCentralWidget(self.wg)
        main_l = QtWidgets.QVBoxLayout(self.wg)

        main_l.addWidget(self.data_widget)

        cam_block = QtWidgets.QVBoxLayout()
        # Angle label
        self.angle_label = QtWidgets.QLabel('Angle: --°')
        font = QtGui.QFont(); font.setPointSize(16); font.setBold(True)
        self.angle_label.setFont(font)
        self.angle_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        cam_block.addWidget(self.angle_label, alignment=QtCore.Qt.AlignLeft)
        # Distance label
        self.distance_label = QtWidgets.QLabel('Distance: -- cm')
        dist_font = QtGui.QFont(); dist_font.setPointSize(16); dist_font.setBold(True)
        self.distance_label.setFont(dist_font)
        self.distance_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        cam_block.addWidget(self.distance_label, alignment=QtCore.Qt.AlignLeft)

        self.cam_label = QtWidgets.QLabel()
        self.cam_label.setFixedSize(w_view, h_view)
        self.cam_label.setScaledContents(True)
        cam_block.addWidget(self.cam_label)
        main_l.addLayout(cam_block)

        ctrl = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel('Frames to record:')
        lbl.setStyleSheet("font-size:16pt;")
        ctrl.addWidget(lbl)
        self.entry = QtWidgets.QLineEdit(); self.entry.setFixedWidth(80); self.entry.setStyleSheet("font-size:16pt;")
        ctrl.addWidget(self.entry)
        ctrl.addStretch()
        self.start_btn = QtWidgets.QPushButton('Start Recording'); self.start_btn.setStyleSheet("font-size:16pt;")
        self.start_btn.clicked.connect(self.startRecording)
        ctrl.addWidget(self.start_btn)
        self.status_label = QtWidgets.QLabel('Ready'); self.status_label.setStyleSheet("font-size:16pt;")
        ctrl.addWidget(self.status_label)
        main_l.addLayout(ctrl)
        self.resize(1025, 1250)

    def updatePlots(self, res: Results):
        data = res['raw_data'].data if self.show_data_type=='raw_data' else res['feature_map'].data
        self.data_widget.setData(data)
        self.last_radar = np.array(data)

    def updateFrame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        raw = frame.copy()
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detector.detectMarkers(gray)
        self.current_angle = np.nan
        self.current_distance = np.nan
        aligned = False

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(raw, corners, ids)
            tvecs = {}
            rvec0 = None

            for corners_i, tid in zip(corners, ids.flatten()):
                if tid in (0, 1, 3):
                    rvecs, tvec_array, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners_i],
                        self.markerLength,
                        self.K,
                        self.dist
                    )
                    tvecs[tid] = tvec_array[0, 0]
                    if tid == 0:
                        rvec0 = rvecs[0, 0]

            if rvec0 is not None and {0, 1, 3}.issubset(tvecs):
                centers = {
                    tid: c.reshape(-1, 2).mean(axis=0)
                    for c, tid in zip(corners, ids.flatten()) if tid in (0, 1)
                }
                u0, v0 = centers[0]
                u1, v1 = centers[1]
                if (abs(u0 - self.cx) < self.align_px_thresh and
                        abs(v1 - v0) < self.baseline_v_thresh):
                    aligned = True

                R_tag2cam, _ = cv2.Rodrigues(rvec0)
                tag_normal = R_tag2cam[:, 2]

                base = tvecs[1] - tvecs[0]
                base_proj = base - (base.dot(tag_normal)) * tag_normal
                base_norm = base_proj / np.linalg.norm(base_proj)

                forward = np.cross(tag_normal, base_norm)
                forward = forward / np.linalg.norm(forward)

                vec03 = tvecs[3] - tvecs[0]
                proj = vec03 - (vec03.dot(tag_normal)) * tag_normal

                dist_m = np.linalg.norm(proj)
                self.current_distance = round(dist_m * 100, 1)

                x = proj.dot(base_norm)
                y = proj.dot(forward)
                angle_rad = np.arctan2(x, y)
                self.current_angle = round(np.degrees(angle_rad), 1)

        angle_text = '--°' if np.isnan(self.current_angle) else f'{self.current_angle:.1f}°'
        dist_text = '-- cm' if np.isnan(self.current_distance) else f'{self.current_distance:.1f} cm'
        self.angle_label.setText(f'Angle: {angle_text}' + ('  [Aligned]' if aligned else ''))
        self.distance_label.setText(f'Distance: {dist_text}')
        self.angles.append(self.current_angle)
        self.distances.append(self.current_distance)

        disp = cv2.flip(raw, 1)
        dh, dw = disp.shape[:2]
        cv2.line(disp, (dw // 2, 0), (dw // 2, dh), (255, 255, 255), 1)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, dw, dh, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).transformed(QtGui.QTransform().scale(-1, 1))
        self.cam_label.setPixmap(pix)

        if self.is_recording and self.recorded_count < self.total_frames and self.last_radar is not None:
            self.video_writer.write(frame)
            self.h5ds[self.recorded_count, ...] = self.last_radar
            self.recorded_count += 1
            self.status_label.setText(f'Recording {self.recorded_count}/{self.total_frames}')
            if self.recorded_count >= self.total_frames:
                self.finishRecording()

    def startRecording(self):
        try:
            n = int(self.entry.text()); assert n>0
        except:
            QtWidgets.QMessageBox.critical(self,'Error','請輸入正整數'); return
        for i in range(3,0,-1):
            self.status_label.setText(f"Recording starts in {i}"); QtWidgets.QApplication.processEvents(); time.sleep(1)
        self.angles = []; self.distances = []
        self.total_frames = n; self.recorded_count = 0; self.is_recording = True
        self.start_btn.setEnabled(False)
        ts = time.strftime('%Y%m%d_%H%M%S')
        base = os.path.join('Record', f'angle_dist_record_{ts}')
        os.makedirs(base, exist_ok=True)
        self.h5file = h5py.File(os.path.join(base, f'data_{ts}.h5'),'w')
        C,H,W = self.last_radar.shape
        self.h5ds = self.h5file.create_dataset('DS1', (n,C,H,W), np.float32)
        self.video_writer = cv2.VideoWriter(
            os.path.join(base,f'video_{ts}.mp4'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            self.desired_fps,(1280,720)
        )
        self.status_label.setText('Recording started')

    def finishRecording(self):
        self.is_recording = False; self.start_btn.setEnabled(True)
        # Interpolate NaNs
        frames = np.arange(len(self.angles))
        ang_arr = np.array(self.angles, dtype=np.float32)
        dist_arr = np.array(self.distances, dtype=np.float32)
        for arr in (ang_arr, dist_arr):
            mask = np.isnan(arr)
            if np.any(mask): arr[mask] = np.interp(frames[mask], frames[~mask], arr[~mask])
        # Write CSV
        csv_path = os.path.join(os.path.dirname(self.h5file.filename), 'records.csv')
        with open(csv_path,'w',newline='') as f:
            w = csv.writer(f); w.writerow(['frame','angle_deg','distance_cm'])
            for i,(a,d) in enumerate(zip(ang_arr, dist_arr)):
                w.writerow([i, f'{a:.1f}', f'{d:.1f}'])
        # Cleanup
        if self.video_writer: self.video_writer.release(); self.video_writer = None
        if self.h5file: self.h5file.close(); self.h5ds = None
        self.status_label.setText('Done Recording')
        QtWidgets.QMessageBox.information(self,'Info',f'Recording saved\nCSV: {csv_path}')

    def closeEvent(self, event):
        FRM.stop(); kgl.ksoclib.closeDevice(); self.cap.release(); event.accept()

if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    kgl.setLib(); connect(); setScript('K60168-Test-00256-008-v0.0.8-20230717_60cm')
    vm = ShowDataViewModel(); win = ShowDataView(vm, show_data_type='feature_map')
    rec = MultiResult4168BReceiver(); setProperty(rec, actions=1, rbank_ch_enable=7, read_interrupt=0, clear_interrupt=0)
    FRM.setReceiver(rec); FRM.setUpdater(vm); FRM.trigger(); FRM.start()
    win.show(); app.exec_()
