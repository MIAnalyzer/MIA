# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:05:05 2020

@author: Koerber
"""

#import PyInstaller.__main__

import subprocess
import sys
def main():
    

# 1111111111111111111111111111111111111111111111111111111111    
#    command = r"python -m PyInstaller --onefile DeepCellDetector.py"
#    add following
#import sys
#sys.setrecursionlimit(5000)
   
    
# 22222222222222222222222222222222222222222222222222222222222
#

    
 
# ☺333333333333333333333333333333333333333333
#from PyInstaller.utils.hooks import collect_submodules   
#hiddenimports=collect_submodules('tensorflow_core')
    

# https://github.com/pyinstaller/pyinstaller/issues/4400
    
    added_files = [
        ('C:/Users/Koerber/.conda/envs/nils_dev/Lib/site-packages/dask/dask.yaml', './dask'),
		('P:/abteilung9/90/nils/DeepCellDetector/icons', 'icons'),
        ('P:/abteilung9/90/nils/DeepCellDetector/icons/*.png', 'icons')
        
        ]
    
    
    
    command2 = r"python -m PyInstaller --noconfirm DeepCellDetector.spec"
    command = r"pyi-makespec -F --noconsole --noupx --log-level=WARN -i ../icons/logo.ico --add-data C:/Users/Koerber/.conda/envs/nils_dev/Lib/site-packages/dask/dask.yaml;./dask --additional-hooks-dir=hooks ../DeepCellDetector.py"
    

    with open('test.log', 'wb') as f: 
        process = subprocess.Popen(command2, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, b''): 
            sys.stdout.write(line.decode(sys.stdout.encoding))
            f.write(line)
        


if __name__ == '__main__':
    main()