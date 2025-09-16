# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:34:20 2022

@author: SFerneyhough
"""

import pandas as pd
import os, glob
import numpy as np
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

folderpath = r'C:\Users\sferneyhough\Desktop\NEW CAP SENSOR UNCERTAINTY\PDS'


file1 = "2022_10_21_CAP.csv"
file2 = '2022_10_21_POSITION.csv'

filea = os.path.join(folderpath, file1)
fileb = os.path.join(folderpath, file2)

rawdf1 = pd.read_csv(filea,low_memory=(False))
rawdf2 = pd.read_csv(fileb,low_memory=(False))

#Convert to datetimes
# rawdf1['ts'] = pd.to_datetime(rawdf1.Date.astype(str) + ' ' + rawdf1['Time Stamp UTC'].astype(str))
rawdf1['ts'] = pd.to_datetime(rawdf1['Protocol Time Stamp(s)'])
rawdf2['ts'] = pd.to_datetime(rawdf2['Timestamp'])

#drop DAYS TO MATCH (if necessary)
rawdf1['ts'] = rawdf1['ts'] + timedelta(hours=7)

# drop unnecessary cols
df1 = rawdf1[['ts',' Data channel 1 (µm)']]
df2 = rawdf2[['ts','AFS-100S PCC-COM8-0x06-Position Counts',
            'AFS-100S PCC-COM8-0x06-Valve Voltage (V)', 'AFS-100S PCC-COM8-0x06-Setpoint (Counts)']].dropna()

# df1['Volt Zero'] = df1['Volt'] - min(df1['Volt'])
# df1['Volt Mean'] = df1.rolling(10)['Volt Zero'].mean()

#COMBINE LASER AND POSITION DATA
df = pd.merge_asof(df2.sort_values(['ts']),df1.sort_values(['ts']),on=['ts'],direction='nearest')

# delete first line
df = df.loc[df['AFS-100S PCC-COM8-0x06-Setpoint (Counts)'] != 109300]
df= df.rename(columns={'AFS-100S PCC-COM8-0x06-Setpoint (Counts)':'Counts SetP','AFS-100S PCC-COM8-0x06-Position Counts':'Counts'})

#Define the labels to replace the ascending and descending points
Pdict = {-1: 'STEP DOWN',1: 'STEP UP'}

diffdf = df.diff()
    
#Indicate if setpoint is rising or falling.
diffdf['Setpoint'] = np.sign(diffdf['Counts SetP'])


#Fill the first row with the correct label
diffdf['Setpoint'].fillna(0, inplace = True)


diffdf['cnt'] = diffdf['Setpoint'].cumsum()

#Assign back into rawdf array
df['P Direction'] = diffdf['Setpoint']
df['cumsum'] = diffdf['cnt']



#need to count cycle iterations, i.e. when counts = 108,000, increase count
df['Cycle #'] = 0
df.loc[(df['cumsum'] == 0) & df['P Direction'] != 0, 'Cycle #'] += 1
df['cycle'] = df['Cycle #'].cumsum()

#replace and forward fill.
df['P Direction'].replace(Pdict, inplace = True, method = 'ffill')
df['P Direction'].replace(0, inplace = True, method = 'ffill')

#test if position counts have settled
df['settled'] = (np.abs((df['Counts']-df['Counts SetP'])) < 3)
# df = df.loc[df.settled,:]

# df['Position [um]'] = df['Volt Mean'] * 50 #capacitance sensor, 50 micron/volt

#last position reading at each setpoint
df_last = df.groupby(['Counts SetP','P Direction','cycle']).last()
df_last = df_last.drop(['Cycle #','cumsum'],axis=1)

df_tail = df.groupby(['Counts SetP','P Direction','cycle'],as_index=False).tail(5)
df_tail_mean = df_tail.groupby(['Counts SetP','P Direction','cycle'],as_index=False).mean()
df_tail_mean = df_tail_mean.loc[df_tail_mean['P Direction'] != 0]

# #convert analog laser voltage into position (manually set in the laser software)
# # df_last['Position [um]'] = df_last['Volt'] * 3.5 + 25 #60um:10V,-10um:-10V
# # df_last['Position [um]'] = df_last['Volt'] * 3.5 + 30 #65um:10V,-5um:-10V




# df_last2 = df_last[(np.abs(stats.zscore(df_last['Position [um]'])) < 1)]


df_std = df_tail_mean.groupby(['Counts SetP','P Direction'])[' Data channel 1 (µm)'].std().dropna()
df_std_combined = df_tail_mean.groupby(['Counts SetP'])[' Data channel 1 (µm)'].std().dropna()

#drop outliers
df_std2 = df_std[(np.abs(stats.zscore(df_std)) < 1)]
df_std_combined2 = df_std_combined[(np.abs(stats.zscore(df_std_combined)) < 1)]

std_mean_nm = df_std2.mean()*1000
std_max_nm = df_std2.max()*1000
std_min_nm = df_std2.min()*1000

std_comb_mean_nm = df_std_combined2.mean()*1000
std_comb_max_nm = df_std_combined2.max()*1000
std_comb_min_nm = df_std_combined2.min()*1000


sns.scatterplot(data=df_tail_mean,x=' Data channel 1 (µm)',y='Counts',style='P Direction',hue='cycle',alpha=0.5)

# #plot
# sns.lineplot(data=df)
# ax2 = plt.twinx()
# sns.lineplot(data=df['Volt'],color='b')

# sns.lineplot(data=df_std2)

# plt.scatter(df_std['Position [um]'],df_std['COM8 - AFS-100S - Setpoint'])

# plt.scatter(df_last['Position [um]'],df_last['AFS-100S PCC-COM8-0x06-Setpoint (Counts)'])

