
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys

if __name__ == "__main__":
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    splash_pix = QPixmap('loading.jpg')
    load_screen = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    load_screen.setMask(splash_pix.mask())
    load_screen.show()

    from DeepCellDetector import DeepCellDetectorUI
    dcdui = DeepCellDetectorUI()
    load_screen.hide()
#    sys.exit(app.exec_())
    app.exec_()