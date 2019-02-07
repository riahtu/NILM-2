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

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.real_appliances.setText(_translate("MainWindow", "TextLabel"))
        self.predicted_appliance.setText(_translate("MainWindow", "TextLabel"))
        self.plotter.setTitle(_translate("MainWindow", "GroupBox"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

