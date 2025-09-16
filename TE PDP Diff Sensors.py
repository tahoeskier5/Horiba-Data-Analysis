# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:55:00 2023

@author: MGrill
"""

import sys

#importing standard libraries
import glob, os
from time import perf_counter
from scipy.linalg import lstsq, qr, pinv, solve, svd, lu_factor, lu_solve, lu
from scipy.optimize import least_squares
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#pull in the PDP data
filepath = r'C:\Users\sferneyhough\Desktop\TESS PDP\\'
filename = r'2023-02-08 TE PDP PTR RAW DATA.csv'
rawdf = pd.read_csv(filepath  + filename)


# Drop NaN's and inf's and reset index
rawdf.dropna(inplace=True)
rawdf.reset_index(drop=True, inplace=True)

# Create function for curve fitting
def createMatrixA(Pref, Voltage, Temp, V_exp, T_exp):
    #Sometimes unstable if V_exp or T_exp is too large, e.g. 4.
    i = len(Pref)
    arr = np.zeros((i, (V_exp)*(T_exp)))
    for x in range(0, i):
        for i in range (0, T_exp):
            for j in range(0, V_exp):
                arr[x, j + i * (V_exp)] = Voltage[x]**j * Temp[x]**i
    ans = lstsq(arr, Pref, cond=None)
    
    
    return ans[0], arr

#Define the labels to replace the ascending and descending points
Pdict = {-1: 'P-',1: 'P+'}
Tdict = {-1: 'T-',1: 'T+'}


# Remove some unneccessary columns
columns_to_drop = ['COM4-2:Ruska 7250i-0x01-1:Temperature-1',
       'COM4-2:Ruska 7250i-0x01-1:Pressure Unit-5',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 01-1',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Temperature 01-7',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 01-3',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Bridge Value 01-5',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Value 01-6',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 02-8',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 02-10',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Bridge Value 02-12',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Value 02-13',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Temperature 02-14',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 03-15',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 03-17',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Bridge Value 03-19',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Value 03-20',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Temperature 03-21',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 04-22',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 04-24',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Bridge Value 04-26',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Value 04-27',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Temperature 04-28',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 05-29',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 05-31',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Bridge Value 05-33',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Value 05-34',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Temperature 05-35',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 06-36',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 06-38',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Bridge Value 06-40',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Value 06-41',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Temperature 06-42',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 07-43',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 07-45',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Bridge Value 07-47',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Value 07-48',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Temperature 07-49',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 08-50',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 08-52',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Bridge Value 08-54',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Reference Value 08-55',
       'COM3-0:PDP 10CH Test Fixture-0x01-1:Temperature 08-56',
       'TCP-2:Espec BTL-433-0x01-1:Humidity Process Value-0',
       'TCP-2:Espec BTL-433-0x01-1:Temperature Process Value-2',
       'COM11-1:Ruska 7250i-0x01-1:Temperature-1',
       'COM11-1:Ruska 7250i-0x01-1:Pressure Unit-5',
       'COM11-1:Ruska 7250i-0x01-1:Program State-9']

columns_to_drop = [col for col in columns_to_drop if col in rawdf.columns]

rawdf = rawdf.drop(columns=columns_to_drop)
                           
                              
    
# Initial Filtering
rawdf = rawdf.loc[rawdf['COM4-2:Ruska 7250i-0x01-1:Program State-9'] == "RUN"]
rawdf = rawdf.drop(columns = ['COM4-2:Ruska 7250i-0x01-1:Program State-9'])
rawdf = rawdf.astype(float)


#get difference in Ruska for differential sensor reading
rawdf['Ruska'] = rawdf['COM4-2:Ruska 7250i-0x01-1:Pressure-0'] - rawdf['COM11-1:Ruska 7250i-0x01-1:Pressure-0']
rawdf['P Setpoint'] = rawdf['COM4-2:Ruska 7250i-0x01-1:Pressure Setpoint-2'] - rawdf['COM11-1:Ruska 7250i-0x01-1:Pressure Setpoint-2']

#rename
rawdf['T Setpoint'] = rawdf['TCP-3:Espec BTL-433-0x01-1:Temperature Set Point-3']


diffdf = rawdf.diff() #pandas diff to find anomalies.
   
    
    
#Indicate if setpoint is rising or falling.
diffdf['Psetpoint'] = np.sign(diffdf['P Setpoint'])
diffdf['Tsetpoint'] = np.sign(diffdf['T Setpoint'])



#Fill the first row with the correct label
diffdf['Psetpoint'].fillna(-1, inplace = True) #0 torr is descending
diffdf['Tsetpoint'].fillna(1, inplace = True) #25C is ascending



#need cycles
diffdf['Cycle #'] = 0

diffdf['cnt'] = diffdf['Psetpoint'].cumsum()
diffdf.loc[(diffdf['cnt'] == -1) & diffdf['Psetpoint'] != 0, 'Cycle #'] += 1


#replace and forward fill.
diffdf['Psetpoint'].replace(Pdict, inplace = True, method = 'ffill')
diffdf['Psetpoint'].replace(0, inplace = True, method = 'ffill')
diffdf['Tsetpoint'].replace(Tdict, inplace = True, method = 'ffill')
diffdf['Tsetpoint'].replace(0, inplace = True, method = 'ffill')


#Assign back into rawdf array
rawdf['P Direction'] = diffdf['Psetpoint']
rawdf['T Direction'] = diffdf['Tsetpoint']

rawdf['cycle'] = diffdf['Cycle #'].cumsum()

# Correct for the bit flip and converting to voltage
adcfs_counts = 8388608
adcref_voltage = 5
maxadc_counts = 2**24

def apply_lambda(x):
    return (x/adcfs_counts)*adcref_voltage if x < adcfs_counts else -((maxadc_counts-x)/adcfs_counts)*adcref_voltage

# def apply_lambda(x):
#     return x+2**24 if x < adcfs_counts else x


columns1 = ['COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 01-2',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 01-4',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 02-9',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 02-11',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 03-16',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 03-18',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 04-23',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 04-25',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 05-30',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 05-32',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 06-37',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 06-39',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 07-44',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 07-46',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 08-51',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 08-53']

for col in columns1:
    rawdf[col] = rawdf[col].apply(apply_lambda)

rawdf.reset_index(drop=True, inplace=True)

# Curve fitting with the zero pressure point
diffpres_column = ['COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 01-2',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 02-9',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 03-16',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 04-23',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 05-30',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 06-37',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 07-44',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Pressure Value 08-51']

diffbridge_column = ['COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 01-4',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 02-11',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 03-18',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 04-25',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 05-32',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 06-39',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 07-46',
'COM9-0:PDP 10CH Test Fixture-0x01-1:Differential Bridge Value 08-53']

    
# def scale_shiz(x):
#     return x/col_max

# for col in columns1:
#     col_max = max(rawdf[col])
#     rawdf[col] = rawdf[col].apply(scale_shiz)
 

##group before curve fitting
df = rawdf.groupby(['T Setpoint','P Setpoint',
                    'P Direction','cycle']).tail(7)
df = df.drop(df.groupby(['T Setpoint','P Setpoint',
                    'P Direction','cycle']).tail(2).index, axis=0)
df_mean = df.groupby(['T Setpoint','P Setpoint',
                    'P Direction','cycle'],as_index=False).mean()
df = df.reset_index()
    
for i in range(8):
    diffpres = df[diffpres_column[i]]
    diffbridge = df[diffbridge_column[i]]
    Coeffs_diff, A_diff = createMatrixA(df['Ruska'], diffpres, diffbridge, 3, 3)
    exec(f"Coeffs_diff{i+1} = Coeffs_diff")
    exec(f"A_diff{i+1} = A_diff")

# Calculate the curve fit output (aka the calibration output)
diff_names = ['diff1', 'diff2', 'diff3', 'diff4', 'diff5', 'diff6', 'diff7', 'diff8']
for i, diff_name in enumerate(diff_names):
    col_diff = diff_name + ' CF Output (PSI)'
    Coeffs = eval("Coeffs_" + diff_name)
    A_all = eval("A_" + diff_name)
    df[col_diff] = np.matmul(A_all, Coeffs)


#calculate the curve fit output's error from reference
cf_columns = ['diff1 CF Output (PSI)', 'diff2 CF Output (PSI)', 'diff3 CF Output (PSI)',
            'diff4 CF Output (PSI)', 'diff5 CF Output (PSI)', 'diff6 CF Output (PSI)', 
            'diff7 CF Output (PSI)', 'diff8 CF Output (PSI)']

df_cf = pd.DataFrame()
df_cf['P SetPoint'] = df['Ruska']
df_cf['T SetPoint'] = df['T Setpoint']
# df_cf['cycle'] = df['cycle']

for col in cf_columns:
    col_error_psi = col + ' Error'
    df[col_error_psi] = df[col] - df['Ruska']
    df_cf[col_error_psi] = df[col_error_psi]

df_cf = df_cf.groupby(['P SetPoint','T SetPoint'],as_index=False).mean()




# Create the report dataframe
reportdf=pd.DataFrame()
reportdf['Relative Time (ms)'] = df['Relative Time (ms)']
reportdf['P SetPoint'] = df['P Setpoint']
reportdf['T SetPoint'] = df['T Setpoint']
reportdf['P Direction'] = df['P Direction']
reportdf['T Direction'] = df['T Direction']
reportdf['cycle'] = df['cycle']
# reportdf['Orientation [deg]'] = df['Orientation [deg]']


# Calculate the sensor inaccuracy
num_accuracy_columns = 8
accuracy_columns = []
for i in range(1, num_accuracy_columns + 1):
    col_name = 'diff{}_inaccuracy (%FS)'.format(i)
    accuracy_columns.append(col_name)
    reportdf[col_name] = (df['diff{} CF Output (PSI)'.format(i)] - df['Ruska']) / 50 * 100






# #drop 3rd sensor from orientation testing- shit was whack
# reportdf = reportdf.drop(columns='diff3_inaccuracy (%FS)')


# Calculate the maximum of the inaccuracies and store the result in the 'inaccuracy_max' column
# reportdf['inaccuracy_max'] = np.diff(reportdf[accuracy_columns].max(axis=1))

# #drop large outliers from orientation data
# reportdf = reportdf.loc[diff(reportdf['diff1_inaccuracy (%FS)']) < 0.2]



reportdf_mean = reportdf.groupby(['P SetPoint','T SetPoint'],as_index=False).mean()

reportdf_mean['inaccuracy_max'] = (np.diff(reportdf_mean[accuracy_columns]).max(axis=1))
# reportdf_mean = reportdf.groupby(['Orientation [deg]'],as_index=False).mean()


reportdf_std = reportdf.groupby(['P SetPoint','P Direction'],as_index=False).std()
reportdf_std['inaccuracy_max'] = reportdf_std[accuracy_columns].max(axis=1)

#ruska sensor noise
ruska_df = df.groupby(['Ruska']).std()


#Separate into pressure directions for hysteresis
df_up = reportdf.loc[reportdf['P Direction'] == 'P+']
df_up = df_up.sort_values(['P SetPoint','T SetPoint'])
df_up = df_up.loc[df_up['P SetPoint'] != 1.5].reset_index()
df_up = df_up.loc[df_up['P SetPoint'] != -1.375]
df_up = df_up.loc[df_up['cycle'] != 4].reset_index()

df_down = reportdf.loc[reportdf['P Direction'] == 'P-']
df_down = df_down.sort_values(['P SetPoint','T SetPoint'])
df_down = df_down.loc[df_down['P SetPoint'] != -1.5].reset_index()
df_down = df_down.loc[df_down['P SetPoint'] != -1.375]
df_down = df_down.loc[df_down['cycle'] != 4].reset_index()

df_hyst = pd.DataFrame()
df_hyst['P SetPoint'] = df_up['P SetPoint']
df_hyst['T SetPoint'] = df_up['T SetPoint']
df_hyst['P Direction'] = df_up['P Direction']
df_hyst['T Direction'] = df_up['T Direction']
df_hyst['cycle'] = df_up['cycle']

df_hyst['diff 1 hyst [%FS]'] = df_up['diff1_inaccuracy (%FS)'] - df_down['diff1_inaccuracy (%FS)']
df_hyst['diff 2 hyst [%FS]'] = df_up['diff2_inaccuracy (%FS)'] - df_down['diff2_inaccuracy (%FS)']
df_hyst['diff 3 hyst [%FS]'] = df_up['diff3_inaccuracy (%FS)'] - df_down['diff3_inaccuracy (%FS)']
df_hyst['diff 4 hyst [%FS]'] = df_up['diff4_inaccuracy (%FS)'] - df_down['diff4_inaccuracy (%FS)']
df_hyst['diff 5 hyst [%FS]'] = df_up['diff5_inaccuracy (%FS)'] - df_down['diff5_inaccuracy (%FS)']
df_hyst['diff 6 hyst [%FS]'] = df_up['diff6_inaccuracy (%FS)'] - df_down['diff6_inaccuracy (%FS)']
df_hyst['diff 7 hyst [%FS]'] = df_up['diff7_inaccuracy (%FS)'] - df_down['diff7_inaccuracy (%FS)']
df_hyst['diff 8 hyst [%FS]'] = df_up['diff8_inaccuracy (%FS)'] - df_down['diff8_inaccuracy (%FS)']

df_hyst_mean = df_hyst.groupby(['P SetPoint','T SetPoint']).mean()


df_std = reportdf.groupby(['P SetPoint','P Direction','T SetPoint','T Direction'],as_index=False).std()
df_std = df_std.rename(columns={'diff1_inaccuracy (%FS)':'diff1_repeatability (%FS)',
                                'diff2_inaccuracy (%FS)':'diff2_repeatability (%FS)',
                                'diff3_inaccuracy (%FS)':'diff3_repeatability (%FS)',
                                'diff4_inaccuracy (%FS)':'diff4_repeatability (%FS)',
                                'diff5_inaccuracy (%FS)':'diff5_repeatability (%FS)',
                                'diff6_inaccuracy (%FS)':'diff6_repeatability (%FS)',
                                'diff7_inaccuracy (%FS)':'diff7_repeatability (%FS)',
                                'diff8_inaccuracy (%FS)':'diff8_repeatability (%FS)',
                                'inaccuracy_max':'Repeatability_max'})






#plots
#inaccuracy plot

# # orientation effect calcs
# Orientation_effect = pd.DataFrame()
# Orientation_effect = reportdf_mean.loc[1:4,:] - reportdf_mean.loc[0,:]


reportdf_mean2 = pd.melt(reportdf_mean,id_vars = ['P SetPoint', 'T SetPoint'],value_vars=['diff1_inaccuracy (%FS)','diff2_inaccuracy (%FS)',
                                                                            'diff3_inaccuracy (%FS)','diff4_inaccuracy (%FS)',
                                                                            'diff5_inaccuracy (%FS)','diff6_inaccuracy (%FS)',
                                                                            'diff7_inaccuracy (%FS)','diff8_inaccuracy (%FS)'])

# 1 psi fs sensors
# reportdf_mean2 = pd.melt(reportdf_mean,id_vars = ['P SetPoint', 'T SetPoint'],value_vars=['diff2_inaccuracy (%FS)',
#                                                                             'diff3_inaccuracy (%FS)','diff4_inaccuracy (%FS)',
#                                                                             'diff5_inaccuracy (%FS)'])

#  1.5 psi fs sensors
# reportdf_mean2 = pd.melt(reportdf_mean,id_vars = ['P SetPoint', 'T SetPoint'],value_vars=['diff1_inaccuracy (%FS)','diff6_inaccuracy (%FS)',
#                                                                             'diff7_inaccuracy (%FS)','diff8_inaccuracy (%FS)'])



sns.set_theme(context='talk',style='whitegrid')
# ax = sns.boxplot(data=reportdf_mean2,x='P SetPoint',y='value',color='blue')

ax = sns.scatterplot(data=reportdf_mean2,x='P SetPoint',y='value',size='T SetPoint',hue='variable')
sns.move_legend(ax,'center left',bbox_to_anchor=(1,1))
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.suptitle('TE PDP Differential Pressure Sensor Inaccuracy')
plt.ylabel('Differential Sensor Inaccuracy [%FS]')
plt.xlabel('Pressure Setpoint [psi]')



# #curve fit plot
# df_cf2 = pd.melt(df_cf,id_vars=['P SetPoint','T SetPoint'])

# sns.set_theme(context='talk',style='whitegrid')

# ax = sns.scatterplot(data=df_cf2,x='P SetPoint',y='value',hue='variable',size='T SetPoint')
# # sns.move_legend(ax,'center left',bbox_to_anchor=(1,1))
# ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
# plt.suptitle('TE PDP Pressure Sensor Curve Fit Error, k=3')
# plt.ylabel('Curve Fit Error [psi]')
# plt.xlabel('Pressure SetPoint [psi]')





# #repeatability plot
# df_std2 = pd.melt(df_std,id_vars=['P SetPoint','T SetPoint'],value_vars=['diff1_repeatability (%FS)',
#                                                                           'diff2_repeatability (%FS)',
#                                                                           'diff3_repeatability (%FS)',
#                                                                           'diff4_repeatability (%FS)',
#                                                                           'diff5_repeatability (%FS)',
#                                                                           'diff6_repeatability (%FS)',
#                                                                           'diff7_repeatability (%FS)',
#                                                                           'diff8_repeatability (%FS)'])
# #remove large repeatability outlier
# # df_std2 = df_std2.loc[df_std2['value'] < 0.1]

# sns.set_theme(context='talk',style='whitegrid')

# # ax = sns.scatterplot(data=df_std2,x='P SetPoint',y='value',hue='variable',size='T SetPoint')

# # ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
# sns.boxplot(data=df_std2,x='P SetPoint',y='value',color='blue')
# plt.suptitle('TE PDP Pressure Sensor Repeatability')
# plt.ylabel('Differential Sensor Repeatability [%FS]')
# plt.xlabel('Pressure SetPoint [psi]')


# #hysteresis plots
# df_hyst = df_hyst.groupby(['P SetPoint','T SetPoint'],as_index=False).mean()
# df_hyst2 = pd.melt(df_hyst,id_vars=['P SetPoint','T SetPoint'],value_vars=['diff 1 hyst [%FS]', 'diff 2 hyst [%FS]', 'diff 3 hyst [%FS]',
#                                                                             'diff 4 hyst [%FS]', 'diff 5 hyst [%FS]', 'diff 6 hyst [%FS]',
#                                                                             'diff 7 hyst [%FS]', 'diff 8 hyst [%FS]'])
# # #remove large outlier
# # df_hyst2 = df_hyst2.loc[df_hyst2['value'] < 0.01]
# sns.set_theme(context='talk',style='whitegrid')

# # ax = sns.scatterplot(data=df_hyst2,x='P SetPoint',y='value',hue='variable',size='T SetPoint')
# # ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
# sns.boxplot(data=df_hyst2,x='P SetPoint',y='value',color='blue')
# plt.suptitle('TE PDP Pressure Sensor Hysteresis')
# plt.ylabel('Differential Sensor Hysteresis [%FS]')
# plt.xlabel('Pressure SetPoint [psi]')










# Plot filtered CF error
# plt.scatter(df['COM11-1:Ruska 7250i-0x01-1:Pressure Setpoint-2'], df['diff1 CF Output (PSI) Error'], label = 'diff CF Output Error (PSI)')
# plt.title('Ruska Overview')
# plt.xlabel('Pressure SetPoint')
# plt.ylabel('Curve Fit Error (PSI)')
# plt.legend()
# plt.show()
  

#calculate statistics on each group.
# statsdf = groupdf.describe(percentiles=[])

# #output files to csv
# diffdf.to_csv(filepath + 'diff_data.csv')
# rawdf.to_csv(filepath + 'raw_data.csv')
# statsdf.to_csv(filepath + 'stats_data.csv')
# reportdf.to_csv(filepath + 'report.csv')