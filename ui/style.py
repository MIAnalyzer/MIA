
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

def setStyle(app):
    # # Force the style to be the same on all OSs:
    app.setStyle("Fusion")
    


    # # Now use a palette to switch to dark colors:
    palette = QPalette()
    palette.setColor(QPalette.Window, getBackgroundColor())
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Disabled, QPalette.WindowText, Qt.gray)
    
    
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, getBrightColor())
    # have no effect, see below
    palette.setColor(QPalette.ToolTipBase, getBackgroundColor())
    palette.setColor(QPalette.ToolTipText, getHighlightColor())

    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, getBackgroundColor())
    
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)


    palette.setColor(QPalette.Highlight,QColor(51,153,255))
    palette.setColor(QPalette.HighlightedText, getHighlightColor())
    
    # every time a tooltip is set, the bg-color resets
    # so background-color here and border do not have an effect
    app.setStyleSheet("""QToolTip { background-color: QColor(53, 53, 53); color: lightblue; border: lightblue solid 1px }""")  
    app.setPalette(palette)
    
    
def getBrightColor(asString = False):
    color = QColor(80, 80, 80)
    if asString:
        return "rgb(%d, %d, %d)" % (color.red(), color.green(), color.blue())
    return color

def getHighlightColor(asString = False):
    color = QColor(173, 216, 230)
    if asString:
        return "rgb(%d, %d, %d)" % (color.red(), color.green(), color.blue())
    return color

def getBackgroundColor(asString = False):
    color = QColor(53, 53, 53)
    if asString:
        return "rgb(%d, %d, %d)" % (color.red(), color.green(), color.blue())
    return color


def styleForm(form):
    # add something like:
    #form.setStyleSheet("QRadioButton { color: red }")
#    palette = QPalette()
#    palette.setColor(QPalette.ToolTipBase, Qt.white)
#    palette.setColor(QPalette.ToolTipText, Qt.white)
#    form.setPalette(palette)
    pass 
