import sys
import os
filepath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(filepath)
os.chdir(filepath)

from startup import start


