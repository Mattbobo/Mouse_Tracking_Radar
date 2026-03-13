import numpy as np
from h5py import File
from PySide2 import QtWidgets, QtCore
from pathlib import Path

class FrameCounter:
    def __init__(self, total_frame):
        self.total_frame = total_frame
        self.current = 0

    def update(self):
        if self.current < self.total_frame:
            self.current += 1
        else:
            self.current = -1
        return self.current

    def init(self):
        self.current = 0

class H5Player(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(parent=None)
        self.setWindowTitle('H5 Player')
        self.resize(1000, 800)
        self.frm_counter = FrameCounter(100)
        self.play_timer = QtCore.QTimer()
        self.play_timer.timeout.connect(self.update)
        self.data = None
        pass

    def setupUI(self):
        wg = QtWidgets.QWidget()
        ly = QtWidgets.QVBoxLayout(wg)
        self.setCentralWidget(wg)
        ly_select_file = QtWidgets.QHBoxLayout()
        ly.addLayout(ly_select_file)

        self.le_select_file = QtWidgets.QLineEdit()
        self.pb_select_file  = QtWidgets.QPushButton('Select file')
        self.pb_select_file.clicked.connect(self.selectFile)
        ly_select_file.addWidget(self.le_select_file)
        ly_select_file.addWidget(self.pb_select_file)

        self.pb_play = QtWidgets.QPushButton('Play')
        self.pb_play.clicked.connect(self.play)
        ly.addWidget(self.pb_play)

        from KKT_Module.KKTGraph.ShowRawData import RawDataGraph
        self.RawData = RawDataGraph()
        # self.RawData.enableWidget(True)
        ly.addWidget(self.RawData)

        self.lb_current_frame = QtWidgets.QLabel('Current frame : 0')
        ly.addWidget(self.lb_current_frame)

        pass

    def selectFile(self):
        if Path('Record').exists():
            Path('Record').mkdir(parents=True, exist_ok=True)
        file, type = QtWidgets.QFileDialog.getOpenFileName(None, caption=f'Select H5 file', dir='Record',
                                                           filter="H5 files(*.h5);;All Files (*.*)")
        if file == '':
            return
        self.le_select_file.setText(file)

        with File(file, 'r') as h5py_file:
            data = h5py_file['DS1'][:]
        data = data * (2**15)
        data = np.transpose(data, (3, 0, 1, 2)).astype('int16')
        data = np.reshape(data, (data.shape[0],data.shape[1], data.shape[2]*data.shape[3]))
        self.data = data
        # self.RawData.CH1.setAxisRange([1, data.shape[2] * data.shape[3] + 200], [-2**15, 2**15])
        # self.RawData.CH2.setAxisRange([1, data.shape[2] * data.shape[3] + 200], [-2 ** 15, 2 ** 15])
        if data.shape[0] == 1:
            self.RawData.setData(data[0][0], data[0][0])
        else:
            self.RawData.setData(data[0][0], data[0][1])


    def play(self):
        self.frm_counter.total_frame = self.data.shape[0]
        self.frm_counter.init()
        self.play_timer.start(40)

    def update(self) -> None:
        current_frame = self.frm_counter.update()
        if current_frame == self.frm_counter.total_frame:
            self.play_timer.stop()
            return
        if self.data.shape[0] == 1:
            self.RawData.setData(self.data[current_frame][0], self.data[current_frame][0])
        else:
            self.RawData.setData(self.data[current_frame][0], self.data[current_frame][1])

        self.lb_current_frame.setText(f'Current frame : {current_frame+1}')



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication([])
    win = H5Player()
    win.setupUI()
    win.show()
    app.exec_()
    pass