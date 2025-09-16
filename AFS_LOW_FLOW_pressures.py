# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 08:59:45 2022

@author: SFerneyhough

Analyzes AFS-100 data and plots errors vs channel Setpoints
"""

from time import perf_counter #timing stuff
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sbs
import os, glob
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

folderpath = r'C:\Users\sferneyhough\Documents\MATLAB\06152022_TEST'
files = glob.glob(os.path.join(folderpath, "500 Sccm Sweep Ch 3 at 1 PER_AFS.csv"))
files2 = glob.glob(os.path.join(folderpath, "500 Sccm Sweep Ch 3 at 1 PER_MFC.csv"))
      
rawdf = pd.concat((pd.read_csv(f) for f in files),ignore_index=True) #concatenates all PER_AFS files into 1 big ass table
rawdf2 = pd.concat((pd.read_csv(f) for f in files2),ignore_index=True) #concatenates all PER_MFC files into 1 big ass table


rawdf2['N2 Eq. Flow'] = (np.round(rawdf2['Flow Reading [REAL]'] / 10)) * 10

# Filter data
df = rawdf.filter(['Time[s]','N2 Eq. Flow',
                            'Ch. 1 Ratio Setpoint','Ch. 1 Actual Ratio Setpoint','Ch. 1 Position Setpoint','Ch. 1 Position',                  
                            'Ch. 2 Ratio Setpoint','Ch. 2 Actual Ratio Setpoint','Ch. 2 Position Setpoint','Ch. 2 Position',
                            'Ch. 3 Ratio Setpoint','Ch. 3 Actual Ratio Setpoint','Ch. 3 Position Setpoint','Ch. 3 Position',
                            'Ch. 4 Ratio Setpoint','Ch. 4 Actual Ratio Setpoint','Ch. 4 Position Setpoint','Ch. 4 Position'])
                            

df2 = rawdf2.filter(['Time[s]','N2 Eq. Flow','Pressure 2 Reading [REAL]'])

# Round N2 Flow
df['N2 Eq. Flow'] = (np.round(df['N2 Eq. Flow'] / 10)) * 10



# Add error columns, Actual SP - Commanded SP
df['Ch. 1 Error'] = df['Ch. 1 Actual Ratio Setpoint'] - df['Ch. 1 Ratio Setpoint']
df['Ch. 2 Error'] = df['Ch. 2 Actual Ratio Setpoint'] - df['Ch. 2 Ratio Setpoint']
df['Ch. 3 Error'] = df['Ch. 3 Actual Ratio Setpoint'] - df['Ch. 3 Ratio Setpoint']
df['Ch. 4 Error'] = df['Ch. 4 Actual Ratio Setpoint'] - df['Ch. 4 Ratio Setpoint']

df['RSS Uncertainty'] = np.sqrt(df['Ch. 1 Error']**2 + df['Ch. 2 Error']**2 + 
                                df['Ch. 3 Error']**2 + df['Ch. 4 Error']**2)

# # Remove outliers greater than 0.05 from Ch. 3 setpoint
# df = df[(np.abs(df['Ch. 3 Actual Ratio Setpoint'] - df['Ch. 3 Ratio Setpoint'])) < 0.1]
# df = df[(np.abs(stats.zscore(df)) < 3)]

# # Group by flowrate, percent
flow_rate = 100,250,500,1000,1500


percent = 1


group_flow = df.loc[df['N2 Eq. Flow'].isin(flow_rate)]
group_flow2 = df2.loc[df2['N2 Eq. Flow'].isin(flow_rate)]

group_flow_per = group_flow.loc[group_flow['Ch. 3 Ratio Setpoint'] == (percent)]



mean_group = group_flow_per.groupby(['N2 Eq. Flow','Ch. 1 Ratio Setpoint',
                                     'Ch. 2 Ratio Setpoint','Ch. 3 Ratio Setpoint',
                                     'Ch. 4 Ratio Setpoint']).mean()

median_group = group_flow_per.groupby(['N2 Eq. Flow','Ch. 1 Ratio Setpoint',
                                       'Ch. 2 Ratio Setpoint','Ch. 3 Ratio Setpoint',
                                       'Ch. 4 Ratio Setpoint'], as_index=False).median() # as_index=False keeps the columns as data after groupby


last_group = group_flow_per.groupby(['N2 Eq. Flow','Ch. 1 Ratio Setpoint',
                                     'Ch. 2 Ratio Setpoint','Ch. 3 Ratio Setpoint',
                                     'Ch. 4 Ratio Setpoint'], as_index=False).last()

# last_group2 = group_flow.groupby(['N2 Eq. Flow','Ch. 1 Ratio Setpoint',
#                                      'Ch. 2 Ratio Setpoint','Ch. 3 Ratio Setpoint',
#                                      'Ch. 4 Ratio Setpoint'], as_index=False).last()
 


  
pressures = group_flow2.loc[group_flow2['Time[s]'].isin(last_group['Time[s]'])]

# pressures_last = pressures.groupby(rawdf2['Time[s]']).last()

# Drop outliers greater than # std. deviations
# last_group = last_group[(np.abs(stats.zscore(last_group['RSS Uncertainty'])) < 3)]



# plots 

# last_group = last_group[(last_group['RSS Uncertainty']) < 10]




# 3D TRI SURFACE PLOT

# def scatter3d(x,y,z,cs, colorsMap='plasma'):
#     cm = plt.get_cmap(colorsMap)
#     cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=0.2) #set maximum value for colorbar
#     scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.scatter(x, y, z, s = last_group['N2 Eq. Flow']/10,c=scalarMap.to_rgba(cs))
#     scalarMap.set_array(cs)
#     fig.colorbar(scalarMap,label='Uncertainty %RDG')
#     ax.set_xlabel('Ch. 1')
#     ax.set_ylabel('Ch. 2')
#     ax.set_zlabel('Ch. 4')
    
    


# x = median_group['Ch. 1 Actual Ratio Setpoint']
# y = median_group['Ch. 2 Actual Ratio Setpoint']
# z = median_group['Ch. 4 Actual Ratio Setpoint']
# zz = median_group['RSS Uncertainty']

# scatter3d(x,y,z,zz)

#drop columns for pairplot
# median_group = median_group.drop(columns=['Ch. 1 Error','Ch. 2 Error','Ch. 3 Error','Ch. 4 Error',
                                      # 'Ch. 1 Actual Ratio Setpoint', 'Ch. 2 Actual Ratio Setpoint','Ch. 3 Actual Ratio Setpoint','Ch. 4 Actual Ratio Setpoint'])


# median_group = median_group[(median_group['RSS Uncertainty']) < 1]

#normalize last group uncertainty



# sbs.pairplot(median_group, hue='RSS Uncertainty', diag_kind=None, palette='plasma')

# # plt.savefig('RSS Uncertainty_' + str(percent) + '.png')
