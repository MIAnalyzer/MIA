# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 16:39:13 2020

@author: Koerber
"""


from ui.ui_Painter import Painter

class NoPainter(Painter):
    def __init__(self, canvas):
        super(NoPainter,self).__init__(canvas)
      

    def load(self):
        pass
        
    def save(self):
        pass
    
    def checkForChanges(self):
        pass