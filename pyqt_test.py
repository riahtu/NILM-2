import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QMainWindow, QGridLayout
from PyQt5.QtCore import QThread, pyqtSignal, QDateTime
import pickle
from typing import Dict, Any
import pandas as pd
from nilmtk import DataSet
import zejian_nilm2 as zz
import time
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


# pyuic5 -x mainwindow.ui -o mainwindow.py
# zejianData\Box Sync\power\simulation\NILM

#创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)
    #第四步：就是画图，【可以在此类中画，也可以在其它类中画】
    def plotsin(self):
        self.axes0 = self.fig.add_subplot(111)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes0.plot(t, s)
    def plotcos(self):
        t = np.arange(0.0, 3.0, 0.01)
        s = np.cos(2 * np.pi * t)
        self.axes.plot(t, s)




# 读入数据组 TODO: 这个数据组改成读入式的
iawe = DataSet('data/iawe.h5')
iawe.set_window(start='6-20-2013', end='6-21-2013')
elec = iawe.buildings[1].elec
# 定义想要检测的用电器
APPLIANCES = ['fridge', 'computer']
start_time = next(elec['fridge'].load(ac_type='active'))['power', 'active'].index[0]
delay_time = '0 days 00:01:00'
tspan = [start_time, start_time + pd.Timedelta(delay_time)]


# 做一个合成的数据
APPLIANCES = ['fridge', 'computer']
events: Dict[Any, Any] = {}
mains: Dict[Any, Any] = {}
start_time = next(elec['fridge'].load(ac_type='active'))['power', 'active'].index[0]
delay_time = '0 days 00:01:00'
tspan = [start_time, start_time + pd.Timedelta(delay_time)]
# 读入神经网络 PNN
pnn = pickle.load(open('params/pnn.txt', "rb"))

# 保存activation num
activ_on_list = [start_time]
activ_off_list = [start_time]

#创建一个buffer
buffer_size = 1 * 20 * 60  # buffer_size is 10 mins


# fill buffer
def get_buffer():
    global APPLIANCES
    global mains_activ, tspan, mains_buffer_reactiv

    # 创建一个buffer
    mains_buffer = pd.Series([])
    mains_buffer_reactiv = pd.Series([])
    for j in range(2, 1600):
        # 读入buffer
        mains_activ = 0
        mains_reactiv = 0
        for i in APPLIANCES:
            # print(elec[i].available_columns())
            mains_activ += next(elec[i].load(ac_type='active'))['power', 'active'].fillna(0)[tspan[0]:tspan[1]]
            mains_reactiv += next(elec[i].load(ac_type='reactive'))['power', 'reactive'].fillna(0)[tspan[0]:tspan[1]]
        # 如果遇到丢掉的数据就用前后来非零来代替 TODO: fix it with a better algorithm to handle package loss
        mains_activ = mains_activ.fillna(method='ffill')
        mains_reactiv = mains_reactiv.fillna(method='ffill')

        # 保存到buffer and remove duplicates
        mains_buffer = mains_buffer.append(mains_activ)
        mains_buffer = mains_buffer[~mains_buffer.index.duplicated()]
        mains_buffer_reactiv = mains_buffer_reactiv.append(mains_reactiv)
        mains_buffer_reactiv = mains_buffer_reactiv[~mains_buffer_reactiv.index.duplicated()]

        # check to see if buffer is ready
        if mains_buffer.size > buffer_size + 1:
            mains_buffer = mains_buffer.drop(mains_buffer.index[0:mains_buffer.size - buffer_size])
            mains_buffer_reactiv = mains_buffer_reactiv.drop(
                mains_buffer_reactiv.index[0:mains_buffer_reactiv.size - buffer_size])

            return mains_buffer, mains_buffer_reactiv
        # 增加时间
        tspan = [start_time + (j - 1) * pd.Timedelta(delay_time), start_time + j * pd.Timedelta(delay_time)]
        # time.sleep(1)

def detect_appliances(on_event_trigger, off_event_trigger, plotter_trigger):
    global plotter
    active_power, reactive_power = get_buffer()
    #plotter.plot_power(active_power)
    #plotter_containner.plotcos()
    #plotter_trigger.emit()
    flag_event = False

    # event detection 和 active/ reactive power 计算
    on_event_active, off_event_active = zz.get_activation(active_power, window1=50)
    # 如果没有event就出去重新来，期间保存好这包数据当作buffer
    if len(on_event_active) == 0 and len(off_event_active) == 0:
        ui.predicted_appliance.setText(str(tspan[0]) + str(tspan[1]) + 'no event')

    if len(on_event_active) == 0:
        on_in = []
    else:
        on_in = on_event_active.index
    if len(off_event_active) == 0:
        off_in = []
    else:
        off_in = off_event_active.index

    on_event_reactive, off_event_reactive = zz.get_others(reactive_power, on_in, off_in)

    # do prediction
    if len(on_event_active) != 0:
        appliance_on = zz.predict_appliance(pnn, on_event_active, on_event_reactive)
    if len(off_event_active) != 0:
        appliance_off = zz.predict_appliance(pnn, off_event_active, off_event_reactive, flag_off=True)
    if len(on_event_active) != 0:
        for k in range(appliance_on.size):
            # 对照以往的activation，得到现在的activation
            activ_on_list.append(on_event_active.index[k])
            if appliance_on[k] == 0:
                on_event = str(on_event_active.index[k]) + '  fridge on'
            elif appliance_on[k] == 2:
                on_event = str(on_event_active.index[k]) + '  computer on'
            elif appliance_on[k] == 1:
                on_event = str(on_event_active.index[k]) + '  cloth iron on'
            on_event_trigger.emit(on_event)
            time.sleep(2)

    if len(off_event_active) != 0:
        for k in range(appliance_off.size):
            # 对照以往的activation，得到现在的activation
            activ_off_list.append(off_event_active.index[k])
            if appliance_off[k] == 0:
                off_event = str(off_event_active.index[k]) + '  fridge off'
            elif appliance_off[k] == 2:
                off_event = str(off_event_active.index[k]) + '  computer off'
            elif appliance_off[k] == 1:
                off_event = str(off_event_active.index[k]) + '  cloth iron off'
            off_event_trigger.emit(off_event)
            time.sleep(2)
            #self.junk_printer.append(off_event)

    # ui.predicted_appliance.setText(data)
    # ui.real_appliances.setText(data)


class WorkThread(QThread):
    on_event_trigger = pyqtSignal(str)
    off_event_trigger = pyqtSignal(str)
    plotter_trigger = pyqtSignal()
    def __int__(self):
        super(WorkThread, self).__init__()

    def run(self):
        while True:
            detect_appliances(self.on_event_trigger, self.off_event_trigger, self.plotter_trigger)



# main window class
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(675, 465)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.real_appliances = QtWidgets.QLabel(self.centralWidget)
        self.real_appliances.setGeometry(QtCore.QRect(10, 0, 361, 21))
        self.real_appliances.setObjectName("real_appliances")
        self.predicted_appliance = QtWidgets.QLabel(self.centralWidget)
        self.predicted_appliance.setGeometry(QtCore.QRect(10, 20, 371, 21))
        self.predicted_appliance.setObjectName("predicted_appliance")
        self.plotter = QtWidgets.QGroupBox(self.centralWidget)
        self.plotter.setGeometry(QtCore.QRect(10, 50, 641, 351))
        self.plotter.setObjectName("plotter")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 675, 23))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        # 第五步：定义MyFigure类的一个实例
        # 绘图器
        self.plotter_containner = MyFigure(width=3, height=2, dpi=100)
        self.plotter_containner.plotsin()
        self.gridlayout = QGridLayout(self.plotter)
        self.gridlayout.addWidget(self.plotter_containner, 0, 1)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.real_appliances.setText(_translate("MainWindow", "TextLabel"))
        self.predicted_appliance.setText(_translate("MainWindow", "TextLabel"))
        self.plotter.setTitle(_translate("MainWindow", "GroupBox"))

    def write_on(self, in_data):
        self.predicted_appliance.setText(in_data)
        self.plotter_containner.plotcos()
        self.plotter.update()
        print('in_on')

    def write_off(self, in_data):
        self.real_appliances.setText(in_data)
        self.plotter_containner.plotcos()
        self.plotter.update()
        print('in_off')

    # 定义画图器来画active power
    def plot_power(self, data):
        #plotter.axes.plot(data.values)
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes.plot(t, s)
        #self.plotter.




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    work = WorkThread()
    # test label
    ui.predicted_appliance.setText('pig')
    ui.real_appliances.setText('shit')
    # test label

    work.off_event_trigger.connect(ui.write_off)
    work.on_event_trigger.connect(ui.write_on)
    #work.plotter_trigger.connect(ui.plot_power)

    work.start()
    window.show()
    sys.exit(app.exec_())