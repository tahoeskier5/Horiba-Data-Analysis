# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 11:57:49 2023

@author: SFerneyhough
"""

import glob, os
from time import perf_counter
from scipy.linalg import lstsq, qr, pinv, solve, svd, lu_factor, lu_solve, lu
from scipy.optimize import least_squares
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#pull in the PDP data
filepath = r'C:\Users\sferneyhough\Desktop\rocker valve laser testing\\'
filename = r'AFS VS ROCKER STROKE.csv'
rawdf = pd.read_csv(filepath  + filename)

# rawdf['mm'] = rawdf['mm'] * 1000
# rawdf = rawdf.rename(columns={'mm':'Test Stand Microns','micron':'Laser Sensor Microns'})
# df_up = rawdf.loc[rawdf['direction'] == 'up']
# df_down = rawdf.loc[rawdf['direction'] == 'down']


rawdf['Laser Sensor [um]'] = abs(rawdf['Laser Sensor [um]'])

sns.set_theme(context='talk',style='whitegrid')
sns.lineplot(data=rawdf,x='Voltage Setpoint [%FS]',y='Laser Sensor [um]',style='Valve')
plt.suptitle('Rocker Valve Stroke Reverser and 2x Multiplier vs AFS')