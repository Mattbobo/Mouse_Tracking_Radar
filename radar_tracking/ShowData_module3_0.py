from KKT_Module import kgl
from KKT_Module.DataReceive.Core import Results
from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
from KKT_Module.FiniteReceiverMachine import FRM
from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc
from KKT_Module.GuiUpdater.GuiUpdater import Updater
from KKT_UI.KKTGraph import ShowADCRaw, ShowFeatureMap
from KKT_UI.QTWidget.MainWindows import KKTMainWindow
from PySide2 import QtWidgets, QtCore


def connect():
    try:
        device = kgl.ksoclib.connectDevice()
        if device == 'Unknow':
            ret = QtWidgets.QMessageBox.warning(None, 'Unknown Device', 'Please reconnect device and try again',
                                                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            if ret == QtWidgets.QMessageBox.Ok:
                connect()

    except Exception as e:

        ret = QtWidgets.QMessageBox.warning(None, 'Connection Failed', 'Please reconnect device and try again',
                                            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        if ret == QtWidgets.QMessageBox.Ok:
            connect()


def setProperty(obj: object, **kwargs):
    print(f'==== Set properties in {obj.__class__.__name__} ====')
    for k, v in kwargs.items():
        if not hasattr(obj, k):
            print(f'Attribute "{k}" not in {obj.__class__.__name__}.')
            continue
        # assert hasattr(self, k), 'Attribute "{}" not in receiver.'.format(k)
        obj.__setattr__(k, v)
        print(f'Attribute "{k}", set "{v}"')

def setScript(setting_name: str):
    ksp = SettingProc()
    setting_config = SettingConfigs()
    setting_config.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
    setting_config.Processes = [
        'Reset Device',
        'Gen Process Script',
        'Gen Param Dict', 'Get Gesture Dict',
        'Set Script',
        # 'Phase Calibration',
        'Run SIC',
        'Phase Calibration',
        'Modulation On'
    ]
    setting_config.setScriptDir(f'{setting_name}')
    ksp.startUp(setting_config)


class ShowDataSentinel(QtCore.QObject):
    sig_update_plot = QtCore.Signal(object)


class ShowDataViewModel(Updater):
    def __init__(self):
        super(ShowDataViewModel, self).__init__()
        self.sentinel = ShowDataSentinel()

    def update(self, res:Results):
        self.sentinel.sig_update_plot.emit(res)


class ShowDataView(KKTMainWindow):
    def __init__(self, view_model: ShowDataViewModel, show_data_type: str = 'raw_data'):
        super(ShowDataView, self).__init__(title='Show Data tool')
        self.vm = view_model
        self.vm.sentinel.sig_update_plot.connect(self.updatePlots)
        self.show_data_type = show_data_type
        if self.show_data_type == 'raw_data':
            kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
        elif self.show_data_type == 'feature_map':
            kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)
        self.data_widget = None
        self.resize(800, 600)

    def setup(self):
        self.wg = QtWidgets.QWidget()
        ly = QtWidgets.QHBoxLayout(self.wg)
        self.setCentralWidget(self.wg)

        if self.show_data_type == 'raw_data':
            self.data_widget = ShowADCRaw.MultiRawDataPLotsWidget()
        elif self.show_data_type == 'feature_map':
            self.data_widget = ShowFeatureMap.MultiFeatureMapPlotsWidget()

        if self.data_widget is not None:
            ly.addWidget(self.data_widget)

    def updatePlots(self, result):
        if self.data_widget is not None:
            if self.show_data_type == 'raw_data':
                result = result['raw_data'].data
            elif self.show_data_type == 'feature_map':
                result = result['feature_map'].data

            self.data_widget.setData(result)


def closeEvent(event):
    FRM.stop()
    kgl.ksoclib.closeDevice()


if __name__ == '__main__':
    setting_file = r'K60168-Test-00256-008-v0.0.8-20230717_120cm'
    app = QtWidgets.QApplication([])

    kgl.setLib()
    # kgl.ksoclib.switchLogMode(True)   # print C# library log
    connect()
    setScript(setting_file)

    view_model = ShowDataViewModel()
    win = ShowDataView(view_model, show_data_type='feature_map')
    win.closeEvent = closeEvent
    win.setup()

    receiver = MultiResult4168BReceiver()
    receiver_args = {
          "actions": 1,
          "rbank_ch_enable": 7,
          "read_interrupt": 0,
          "clear_interrupt": 0
        }
    setProperty(receiver, **receiver_args)
    FRM.setReceiver(receiver)
    FRM.setUpdater(view_model)
    FRM.trigger()
    FRM.start()

    win.show()

    app.exec_()
