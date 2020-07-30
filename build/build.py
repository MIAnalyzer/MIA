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
#added_files = [
#         ('C:/Users/Koerber/.conda/envs/nils_dev_2/Lib/site-packages/dask/dask.yaml', './dask'),
# 		('C:/cudnn-10.1-windows10-x64-v7.6.3.30/cuda/bin/cudnn64_7.dll', '.'),
# 		('./add/msvcp140.dll', '.'),
# 		('./add/msvcp140_1.dll', '.')
#         ]

    
 
# ☺333333333333333333333333333333333333333333
#from PyInstaller.utils.hooks import collect_submodules   
#hiddenimports=collect_submodules('tensorflow_core')


# 44444444444444444444444444444
# datas=added_files,
# hiddenimports=collect_submodules('tensorflow_core'),
# hookspath=['hooks'],

# 5555555555555555555
# copy / add google_api_python_client-xxx
    

    
    added_files = [
        ('C:/Users/Koerber/.conda/envs/nils_dev/Lib/site-packages/dask/dask.yaml', './dask'),
		('P:/abteilung9/90/nils/DeepCellDetector/icons', 'icons'),
        ('P:/abteilung9/90/nils/DeepCellDetector/icons/*.png', 'icons')
        ]
    
    
    
    build_exe = r"python -m PyInstaller --noconfirm startup.spec"
    spec = r"pyi-makespec -F --noconsole --noupx --log-level=WARN -i ../icons/logo.ico --additional-hooks-dir=hooks ../startup.py"
    spec_debug = r"pyi-makespec -D --noupx --debug all --log-level=DEBUG ../startup.py"
    spec_test = r"pyi-makespec -D --noupx --debug all --log-level=DEBUG testbuild.py"
    build_test = r"python -m PyInstaller --noconfirm testbuild.spec"

    with open('test.log', 'wb') as f: 
        process = subprocess.Popen(build_exe, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, b''): 
            sys.stdout.write(line.decode(sys.stdout.encoding))
            f.write(line)


if __name__ == '__main__':
    main()