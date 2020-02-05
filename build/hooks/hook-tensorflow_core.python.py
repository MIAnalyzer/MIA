# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 13:35:54 2020

@author: Koerber
"""



from PyInstaller.utils.hooks import collect_submodules, collect_data_files

hiddenimports = collect_submodules('tensorflow_core')
datas = collect_data_files('tensorflow_core', subdir=None, include_py_files=True)