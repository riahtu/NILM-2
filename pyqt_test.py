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
# zejianData\Box Sync\power\simulation\NILM\GUI\nilm
# lab_stuff\Box Sync\power\simulation\NILM\GUI\nilm

#创建一个matplotlib图形绘制类
class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 第二步：在父类中激活Figure窗口
        super(MyFigure,self).__init__(self.fig) #此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)
    # debug plot
    def plotsin(self):
        self.axes0 = self.fig.add_subplot(111)
        self.axes0.clear()
        t = np.arange(0.0, 3.0, 0.01)
        s = np.sin(2 * np.pi * t)
        self.axes0.plot(t, s)
        self.draw()
    def plotcos(self):
        self.axes.clear()
        t = np.arange(0.0, 3.0, 0.01)
        s = np.cos(2 * np.pi * t)
        self.axes.plot(t, s)
        self.draw()
    # debug plot
    def plotpp(self, tp):
        self.axes.clear()
        self.axes.plot(active_power.values)
        self.draw()




# 读入数据组 TODO: 这个数据组改成读入式的
iawe = DataSet('data/iawe.h5')
iawe.set_window(start='6-20-2013', end='6-21-2013')
elec = iawe.buildings[1].elec
# 定义想要检测的用电器
APPLIANCES = ['fridge', 'computer']
# APPLIANCES = ['fridge']
start_time = next(elec['fridge'].load(ac_type='active'))['power', 'active'].index[0]
delay_time = '0 days 00:01:00'
tspan = [start_time, start_time + pd.Timedelta(delay_time)]

# 读入神经网络 PNN
pnn = pickle.load(open('params/pnn.txt', "rb"))

# 保存结果
f_on = open('res/detect_app_on.txt', 'w')
f_off = open('res/detect_app_off.txt', 'w')

# 保存activation num
activ_on_list = [start_time]
activ_off_list = [start_time]

#创建一个buffer
buffer_size = 1 * 20 * 60  # buffer_size is 10 mins
active_power=0

plotter_containner = MyFigure(width=3, height=2, dpi=100)

# this is used for restarting the work thread
flag_restart = False


class WorkThread(QThread):
    on_event_trigger = pyqtSignal(str)
    off_event_trigger = pyqtSignal(str)
    plotter_trigger = pyqtSignal()
    img_change_trigger = pyqtSignal(int)
    flag_buffer_ready = False
    current_last_event_on = start_time
    current_last_event_off = start_time
    def __int__(self):
        super(WorkThread, self).__init__()
        self.flag = True # thread stop flag

    def run(self):
        self.flag = True
        while self.flag:
            self.detect_appliances(self.on_event_trigger, self.off_event_trigger, self.plotter_trigger)
        print('work thread closed')

    def stop(self):
        global f_off, f_on
        print("prepare to stop work thread")
        self.flag = False
        self.flag_buffer_ready = False
        print(self.flag)
        f_off.close()
        f_on.close()

    # fill buffer
    def get_buffer(self):
        global APPLIANCES
        global mains_activ, tspan, mains_buffer_reactiv, start_time

        # 创建一个buffer
        mains_buffer = pd.Series([])
        mains_buffer_reactiv = pd.Series([])
        for j in range(2, 1600):
            # 检测是否需要再读入buffer
            if flag_restart:
                break
            # 读入buffer
            mains_activ = 0
            mains_reactiv = 0
            # 这里直接相加有个问题，如果其中一段数据没有的话就会加成NAN
            for i in APPLIANCES:
                # print(elec[i].available_columns())
                if type(mains_activ) == type(0):
                    mains_temp = mains_activ
                else:
                    mains_temp = mains_activ.copy()
                mains_activ += next(elec[i].load(ac_type='active'))['power', 'active'].fillna(0)[tspan[0]:tspan[1]]
                adder = -(mains_temp - mains_activ)
                mains_activ = mains_temp + adder.fillna(0)

                if type(mains_reactiv) == type(0):
                    mains_temp = mains_reactiv
                else:
                    mains_temp = mains_reactiv.copy()
                mains_reactiv += next(elec[i].load(ac_type='reactive'))['power', 'reactive'].fillna(0)[
                                 tspan[0]:tspan[1]]
                adder = -(mains_temp - mains_reactiv)
                mains_reactiv = mains_temp + adder.fillna(0)
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
                # 增加时间
                start_time = start_time + (j - 1) * pd.Timedelta(delay_time)
                return mains_buffer, mains_buffer_reactiv
            # 增加时间
            tspan = [start_time + (j - 1) * pd.Timedelta(delay_time), start_time + j * pd.Timedelta(delay_time)]
            # time.sleep(1)

    def get_one_reading(self, mains_buffer, mains_buffer_reactiv):
        global APPLIANCES
        global tspan, start_time

        # 读入buffer
        mains_activ = 0
        mains_reactiv = 0
        # 这里直接相加有个问题，如果其中一段数据没有的话就会加成NAN
        for i in APPLIANCES:
            # print(elec[i].available_columns())
            if type(mains_activ) == type(0):
                mains_temp = mains_activ
            else:
                mains_temp = mains_activ.copy()
            mains_activ += next(elec[i].load(ac_type='active'))['power', 'active'].fillna(0)[tspan[0]:tspan[1]]
            adder = -(mains_temp - mains_activ)
            mains_activ = mains_temp + adder.fillna(0)

            if type(mains_reactiv) == type(0):
                mains_temp = mains_reactiv
            else:
                mains_temp = mains_reactiv.copy()
            mains_reactiv += next(elec[i].load(ac_type='reactive'))['power', 'reactive'].fillna(0)[
                             tspan[0]:tspan[1]]
            adder = -(mains_temp - mains_reactiv)
            mains_reactiv = mains_temp + adder.fillna(0)
        # 如果遇到丢掉的数据就用前后来非零来代替 TODO: fix it with a better algorithm to handle package loss
        mains_activ = mains_activ.fillna(method='ffill')
        mains_reactiv = mains_reactiv.fillna(method='ffill')

        # 保存到buffer and remove duplicates
        mains_buffer = mains_buffer.append(mains_activ)
        mains_buffer = mains_buffer[~mains_buffer.index.duplicated()]
        mains_buffer_reactiv = mains_buffer_reactiv.append(mains_reactiv)
        mains_buffer_reactiv = mains_buffer_reactiv[~mains_buffer_reactiv.index.duplicated()]

        # check to see if buffer is ready
        mains_buffer = mains_buffer.drop(mains_buffer.index[0:mains_buffer.size-buffer_size])
        mains_buffer_reactiv = mains_buffer_reactiv.drop(mains_buffer_reactiv.index[0:mains_buffer_reactiv.size-buffer_size])
        # 增加时间
        start_time = start_time + pd.Timedelta(delay_time)
        tspan = [start_time, start_time + pd.Timedelta(delay_time)]
        return mains_buffer, mains_buffer_reactiv

    def detect_appliances(self, on_event_trigger, off_event_trigger, plotter_trigger):
        global active_power, plotter_containner, reactive_power
        if not self.flag_buffer_ready:
            active_power, reactive_power = self.get_buffer()
            self.flag_buffer_ready = True
        else:
            active_power, reactive_power = self.get_one_reading(active_power, reactive_power)

        # plotter.plot_power(active_power)
        # plotter_containner.plot_power(active_power)
        # plotter_trigger.emit()
        # plotter_containner.axes.clear()
        # plotter_containner.axes.plot(active_power.values)
        # plotter_containner.axes.draw()
        plotter_trigger.emit()
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
                # check if this is the latest event
                if on_event_active.index[k] > self.current_last_event_on:
                    self.current_last_event_on = on_event_active.index[k]
                    print(on_event, file=f_on)
                    on_event_trigger.emit(on_event)
                    if appliance_on[k] == 0:
                        self.img_change_trigger.emit(11)
                    elif appliance_on[k] == 2:
                        self.img_change_trigger.emit(21)
                    time.sleep(0.2)

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
                # check if this is the latest event
                if off_event_active.index[k] > self.current_last_event_off:
                    self.current_last_event_off = off_event_active.index[k]
                    off_event_trigger.emit(off_event)
                    print(off_event, file=f_off)
                    if appliance_off[k] == 0:
                        self.img_change_trigger.emit(10)
                    elif appliance_off[k] == 2:
                        self.img_change_trigger.emit(20)
                    time.sleep(0.2)
                # self.junk_printer.append(off_event)
        time.sleep(0.2)

# main window class
class Ui_MainWindow(object):
    stop_thread_trigger = pyqtSignal()
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(644, 445)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.fix_time_interval = QtWidgets.QLabel(self.centralWidget)
        self.fix_time_interval.setGeometry(QtCore.QRect(10, 10, 361, 21))
        self.fix_time_interval.setObjectName("fix_time_interval")
        self.start_time = QtWidgets.QDateEdit(self.centralWidget)
        self.start_time.setGeometry(QtCore.QRect(10, 40, 110, 22))
        self.start_time.setDateTime(QtCore.QDateTime(QtCore.QDate(2013, 6, 20), QtCore.QTime(0, 0, 0)))
        self.start_time.setObjectName("start_time")
        self.fix_to = QtWidgets.QLabel(self.centralWidget)
        self.fix_to.setGeometry(QtCore.QRect(130, 40, 21, 21))
        self.fix_to.setObjectName("fix_to")
        self.end_time = QtWidgets.QDateEdit(self.centralWidget)
        self.end_time.setGeometry(QtCore.QRect(150, 40, 110, 22))
        self.end_time.setDateTime(QtCore.QDateTime(QtCore.QDate(2013, 6, 21), QtCore.QTime(0, 0, 0)))
        self.end_time.setObjectName("end_time")
        self.button_start = QtWidgets.QPushButton(self.centralWidget)
        self.button_start.setGeometry(QtCore.QRect(280, 40, 41, 23))
        self.button_start.setObjectName("button_start")
        self.plotter = QtWidgets.QGroupBox(self.centralWidget)
        self.plotter.setGeometry(QtCore.QRect(0, 210, 631, 171))
        self.plotter.setObjectName("plotter")
        self.activate_display = QtWidgets.QTextBrowser(self.centralWidget)
        self.activate_display.setGeometry(QtCore.QRect(5, 70, 271, 131))
        self.activate_display.setObjectName("activate_display")
        self.button_stop = QtWidgets.QPushButton(self.centralWidget)
        self.button_stop.setGeometry(QtCore.QRect(330, 40, 41, 23))
        self.button_stop.setObjectName("button_stop")
        self.fridge = QtWidgets.QLabel(self.centralWidget)
        self.fridge.setGeometry(QtCore.QRect(310, 70, 121, 91))
        self.fridge.setObjectName("fridge")
        self.computer = QtWidgets.QLabel(self.centralWidget)
        self.computer.setGeometry(QtCore.QRect(480, 70, 131, 91))
        self.computer.setObjectName("computer")
        self.fridge_sta = QtWidgets.QLabel(self.centralWidget)
        self.fridge_sta.setGeometry(QtCore.QRect(310, 170, 91, 31))
        self.fridge_sta.setObjectName("fridge_sta")
        self.computer_sta = QtWidgets.QLabel(self.centralWidget)
        self.computer_sta.setGeometry(QtCore.QRect(480, 170, 91, 31))
        self.computer_sta.setObjectName("computer_sta")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 644, 23))
        self.menuBar.setObjectName("menuBar")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)


        # 第五步：定义MyFigure类的一个实例
        self.gridlayout = QGridLayout(self.plotter)
        self.gridlayout.addWidget(plotter_containner, 0, 1)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("NILMTK demo")
        self.fix_time_interval.setText("Select usage time interval   2013/6/20 - 2013/6/30")
        self.fix_to.setText("to")
        self.button_start.setText("Start")
        self.button_stop.setText('Stop')
        self.plotter.setTitle("Active power plot")

        self.img_fridge = QtGui.QPixmap('resource/fridge.jpg')
        self.fridge.setPixmap(self.img_fridge)
        self.img_computer = QtGui.QPixmap('resource/computer.jpeg')
        self.computer.setPixmap(self.img_computer)
        self.img_on = QtGui.QPixmap('resource/on.png')
        self.img_off = QtGui.QPixmap('resource/off.jpg')
        self.fridge_sta.setPixmap(self.img_off)
        self.computer_sta.setPixmap(self.img_off)

        self.work = WorkThread()

        self.work.off_event_trigger.connect(self.write_off)
        self.work.on_event_trigger.connect(self.write_on)
        self.work.plotter_trigger.connect(self.plot_power)
        self.work.img_change_trigger.connect(self.change_appliance_sta)

        # 信号槽定义
        self.button_start.clicked.connect(self.start_work)
        self.button_stop.clicked.connect(self.stop_work)
        #self.stop_thread_trigger.connect(self.work.stop)

    def write_on(self, in_data):
        # self.predicted_appliance.setText(in_data)
        self.activate_display.append(in_data)
        #plotter_containner.plotpp(1)
        print('in_on')

    def write_off(self, in_data):
        # self.real_appliances.setText(in_data)
        self.activate_display.append(in_data)
        #plotter_containner.plotpp(2)
        print('in_off')

    def plot_power(self):
        plotter_containner.plotpp(3)

    def start_work(self):
        global tspan, start_time, delay_time, elec
        start_str_time= "{0}{1}{2}{3}{4}".format(str(self.start_time.date().month()), '-', str(self.start_time.date().day()),
                                           '-', str(self.start_time.date().year()))
        end_str_time = "{0}{1}{2}{3}{4}".format(str(self.end_time.date().month()), '-', str(self.end_time.date().day()),
                                            '-', str(self.end_time.date().year()))
        iawe.set_window(start=start_str_time, end=end_str_time)
        elec = iawe.buildings[1].elec
        # 定义想要检测的用电器
        start_time = next(elec['fridge'].load(ac_type='active'))['power', 'active'].index[0]
        delay_time = '0 days 00:01:00'
        tspan = [start_time, start_time + pd.Timedelta(delay_time)]

        self.work.start()

    def stop_work(self):
        #self.stop_thread_trigger.emit()
        self.work.stop()

    def change_appliance_sta(self, in_data):
        if in_data == 10:
            self.fridge_sta.setPixmap(self.img_off)
        elif in_data == 11:
            self.fridge_sta.setPixmap(self.img_on)
        elif in_data == 20:
            self.computer_sta.setPixmap(self.img_off)
        elif in_data == 20:
            self.computer_sta.setPixmap(self.img_on)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())