# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:23:19 2022

@author: SFerneyhough
"""

import pandas as pd
import os, glob
import numpy as np
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

folderpath = r'C:\Users\sferneyhough\Desktop\cone valve flow repeat'


file1 = "2022-10-17 Cone Valve 2 Flow Repeatability Test 1 (10SCCM).csv"

# file1 = glob.glob(os.path.join(folderpath, "*.csv"))

filea = os.path.join(folderpath, file1)

df = pd.read_csv(filea,encoding='utf-8')
# df = pd.concat((pd.read_csv(f) for f in file1),ignore_index=True)
# df1 = pd.read_csv(file1[0]).dropna(axis=1)
# df1['Valve #'] = 1
# df2 = pd.read_csv(file1[1]).dropna(axis=1)
# df2['Valve #'] = 4
# df3 = pd.read_csv(file1[2]).dropna(axis=1)
# df3['Valve #'] = 4
# df4 = pd.read_csv(file1[3]).dropna(axis=1)
# df4['Valve #'] = 2

# df = pd.concat([df1,df2,df3,df4])

# setpoints
sp_1 = {197000,192190,187380,182570,177760,172950,168140,163330,158530,153710,148900}
sp_2 = {65500,60770,56040,51310,46580,41850,37120,32390,27660,22930,18200}
sp_4 = {170000,163900,157800,151700,145600,139500,133400,127300,121200,115100,109000}
    

df['SetP Rounded'] = np.round(df['AFS-100S PCC-COM15-0x06-Position Counts']/10,0) * 10

df = df.loc[df['SetP Rounded'].isin(sp_2)]


df['Valve #'] = 2
Valve = 'Valve 2'

df = df.rename(columns={'AFS-100S PCC-COM15-0x06-Position Counts':'Counts','SEC-Z512MGX-COM11-0x01-Flow (%)':'Flow [slm]'})
df['Flow [sccm]']  = df['Flow [slm]'] * 1


#Define the labels to replace the ascending and descending points
Pdict = {-1: 'STEP DOWN',1: 'STEP UP'}

diffdf = df.diff()
    
#Indicate if setpoint is rising or falling.
diffdf['Setpoint'] = np.sign(diffdf['SetP Rounded'])


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

# #test if position counts have settled
df['settled'] = (np.abs((df['Counts']-df['SetP Rounded'])) < 3)
df = df.loc[df.settled,:]

df = df.loc[df['cycle'] != 0]

# #last position reading at each setpoint
df_last = df.groupby(['SetP Rounded','cycle','P Direction'],as_index=False).last()
df_last = df_last.drop(['Cycle #','cumsum'],axis=1)

df_tail = df.groupby(['SetP Rounded','cycle','P Direction'],as_index=False).tail(5)
df_tail_mean = df_tail.groupby(['SetP Rounded','cycle','P Direction'],as_index=False).mean()

df_std_last = df_last.groupby(['SetP Rounded','P Direction'])['Flow [sccm]'].std().dropna()
df_std_tail = df_tail_mean.groupby(['SetP Rounded','P Direction'])['Flow [sccm]'].std().dropna()


sns.scatterplot(data=df_tail_mean,x='Flow [sccm]',y='Counts',style='P Direction', alpha=0.5)
plt.suptitle(Valve)

plt.savefig(filea + '.png',bbox_inches='tight')
df_tail_mean.to_csv(filea + ' df_tail_mean.csv')
df_std_tail.to_csv(filea + ' df_std_tail.csv')
