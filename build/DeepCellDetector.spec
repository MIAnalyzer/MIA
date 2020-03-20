# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import sys
sys.setrecursionlimit(5000)
from PyInstaller.utils.hooks import collect_submodules   
#hiddenimports=collect_submodules('tensorflow_core')

added_files = [
        ('C:/Users/Koerber/.conda/envs/nils_dev/Lib/site-packages/dask/dask.yaml', './dask'),
		('C:/cudnn-10.1-windows10-x64-v7.6.3.30/cuda/bin/cudnn64_7.dll', '.'),
		('./add/msvcp140.dll', '.'),
		('./add/msvcp140_1.dll', '.')
#		('P:/abteilung9/90/nils/DeepCellDetector/icons', 'icons'),
#		('P:/abteilung9/90/nils/DeepCellDetector/icons/*.png', './icons')
        ]


a = Analysis(['..\\DeepCellDetector.py'],
             pathex=['P:\\abteilung9\\90\\nils\\DeepCellDetector\\build'],
             binaries=[],
             datas=added_files,
             hiddenimports=collect_submodules('tensorflow_core'),
             hookspath=['hooks'],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='DeepCellDetector',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False , icon='..\\icons\\logo.ico')
