
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
from ui.style import setStyle

import platform

if platform.system() == 'Windows':
	import ctypes
	appid = u'mia_application'
	ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)

def start():
    app = QCoreApplication.instance()
    # app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    setStyle(app)
    splash_pix = QPixmap('icons/loading.png')
    load_screen = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    load_screen.setMask(splash_pix.mask())
    load_screen.show()

    from mia_gui import MIA_UI
    dcdui = MIA_UI()

    load_screen.hide()
#    sys.exit(app.exec_())
    app.exec_()


 

if __name__ == "__main__":
    start()


