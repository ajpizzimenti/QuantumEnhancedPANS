from pathlib import Path
import sys, os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""Directories"""
scriptPath = Path.cwd()
parentPath = scriptPath.parent
dataPath = parentPath / 'data'
srcPath = parentPath / 'src'
modelPath = parentPath / 'models'
plotPath = parentPath / 'plots'
miscPath = parentPath / 'misc'
savePath = parentPath / 'results'
sys.path.append(srcPath.as_posix())

"""Color palette"""
c_orange = '#e69f00'
c_blue = '#0072b2'
c_red = '#d73027'
c_grey = '#999999'
c_green = '#228b22'
c_purple = '#756bb1'
pal_grey = sns.dark_palette(c_grey,10)[::-1]

"""Plot configurations"""
plt.style.use(miscPath / 'plot.mplstyle')