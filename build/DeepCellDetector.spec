# -*- mode: python ; coding: utf-8 -*-

block_cipher = None
import sys
sys.setrecursionlimit(5000)

added_files = [
        ('./add/google_api_python_client-1.8.2.dist-info/*.*', './google_api_python_client-1.8.2.dist-info')
		]

a = Analysis(['..\\startup.py'],
             pathex=['C:\\Users\\User\\Documents\\deep learning\\bfr\\DeepCellDetector\\build'],
             binaries=[],
             datas=added_files,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['torch'],
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
