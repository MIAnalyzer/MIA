# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 10:45:11 2020

@author: Koerber
"""
import subprocess
import sys

def main():

    command2 = r"python -m PyInstaller --noconfirm ImageStitcher.spec"
    command = r"pyi-makespec -F -i ./stitch_icon.ico --noconsole --noupx --log-level=WARN ImageStitcher.py"

    with open('test.log', 'wb') as f: 
        process = subprocess.Popen(command2, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
        for line in iter(process.stdout.readline, b''):  
            sys.stdout.write(line.decode(sys.stdout.encoding))
            f.write(line)


if __name__ == '__main__':
    main()