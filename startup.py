
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from ui.style import setStyle



def start():

    app = QCoreApplication.instance()
    # app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    setStyle(app)
    splash_pix = QPixmap('icons/loading.jpg')
    load_screen = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    load_screen.setMask(splash_pix.mask())
    load_screen.show()
    try:
        from DeepCellDetector import DeepCellDetectorUI
        dcdui = DeepCellDetectorUI()
    except:
        e = sys.exc_info()[0]
        print( "<p>error: %s</p>" % e )
    load_screen.hide()
#    sys.exit(app.exec_())
    app.exec_()



if __name__ == "__main__":
    start()



