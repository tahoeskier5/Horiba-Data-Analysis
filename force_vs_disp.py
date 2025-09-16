# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 14:33:46 2022

@author: SFerneyhough
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, glob
import seaborn as sbs


folderpath = r'C:\Users\sferneyhough\Desktop\Python Codes\\'

file = "force_vs_disp2.csv"

files = os.path.join(folderpath, file)

rawdf = pd.read_csv(files)

# split into integers
volts = range(16)

df = rawdf.loc[rawdf['VDC [V]'].isin(volts)]

df_K = df.loc[df['Material'] == 'K-M45']
df_SS = df.loc[df['Material'] == '1018 SS']



fig, axes = plt.subplots(3,2,sharey=(True),sharex=(True))

sbs.scatterplot(data=df_K,x='Offset [in]',y='Tip #1',hue='VDC [V]',legend=False,ax=axes[0,0])
sbs.lineplot(data=df_K,x='Offset [in]',y='Tip #1',hue='VDC [V]',ax=axes[0,0])
axes[0,0].set_title('K-M45')

sbs.scatterplot(data=df_SS,x='Offset [in]',y='Tip #1',hue='VDC [V]',legend=False,ax=axes[0,1])
sbs.lineplot(data=df_SS,x='Offset [in]',y='Tip #1',hue='VDC [V]',ax=axes[0,1])
axes[0,1].set_title('1018 SS')

sbs.scatterplot(data=df_K,x='Offset [in]',y='Tip #2',hue='VDC [V]',legend=False,ax=axes[1,0])
sbs.lineplot(data=df_K,x='Offset [in]',y='Tip #2',hue='VDC [V]',ax=axes[1,0])

sbs.scatterplot(data=df_SS,x='Offset [in]',y='Tip #2',hue='VDC [V]',legend=False,ax=axes[1,1])
sbs.lineplot(data=df_SS,x='Offset [in]',y='Tip #2',hue='VDC [V]',ax=axes[1,1])

sbs.scatterplot(data=df_K,x='Offset [in]',y='Tip #3',hue='VDC [V]',legend=False,ax=axes[2,0])
sbs.lineplot(data=df_K,x='Offset [in]',y='Tip #3',hue='VDC [V]',ax=axes[2,0])

sbs.scatterplot(data=df_SS,x='Offset [in]',y='Tip #3',hue='VDC [V]',legend=False,ax=axes[2,1])
sbs.lineplot(data=df_SS,x='Offset [in]',y='Tip #3',hue='VDC [V]',ax=axes[2,1])


fig.suptitle('Solenoid Force vs Displacement, Varying Voltage, 9V Coil')
fig.text(0.06,0.5,'Force [lb]',ha='center',va='center',rotation='vertical')
plt.yticks(np.arange(0,0.8,0.1))

