# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
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

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.fix_time_interval.setText(_translate("MainWindow", "Select usage time interval   2013/6/20 - 2013/6/30"))
        self.fix_to.setText(_translate("MainWindow", "to"))
        self.button_start.setText(_translate("MainWindow", "Start"))
        self.plotter.setTitle(_translate("MainWindow", "Active power plot"))
        self.button_stop.setText(_translate("MainWindow", "Stop"))
        self.fridge.setText(_translate("MainWindow", "TextLabel"))
        self.computer.setText(_translate("MainWindow", "TextLabel"))
        self.fridge_sta.setText(_translate("MainWindow", "TextLabel"))
        self.computer_sta.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

