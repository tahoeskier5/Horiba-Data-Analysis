# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:41:20 2022

@author: SFerneyhough
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

folderpath = r'C:\Users\sferneyhough\Desktop\solenoid rocker test'
file = os.path.join(folderpath,'2022_10_25_top_spring_only.csv')

rawdf = pd.read_csv(file).dropna()
rawdf = rawdf.loc[rawdf['Micrometer [mm]'] != 0]

rawdf['Spring Type'] = rawdf['Top Spring'].astype(str) + ' ' + rawdf['Bottom Spring'].astype(str)
rawdf['Micrometer [um]'] = rawdf['Micrometer [mm]'] * 1000

# rawdf.to_csv('pull_force_results.csv')

# df_1 = rawdf.loc[rawdf['Spring Type'] == '0.016 0.032']
# df_2 = rawdf.loc[rawdf['Spring Type'] == '0.025 0.025']
# df_3 = rawdf.loc[rawdf['Spring Type'] == '0.016 0.016']
# df_4 = rawdf.loc[rawdf['Spring Type'] == '0.016 0.025']
# df_5 = rawdf.loc[rawdf['Spring Type'] == '0.032 0.032']


# fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,sharex=(True),sharey=(True))
# fig.suptitle('Solenoid Rocker Pull Testing')
# ax1.set_title('0.016 0.032')
# ax2.set_title('0.025 0.025')
# ax3.set_title('0.016 0.016')
# ax4.set_title('0.016 0.025')
# ax5.set_title('0.032 0.032')

# sns.scatterplot(data=df_1,x='Laser [um]',y='Micrometer [um]',hue='Force [lbf]',size='Run #',sizes=(50,250),palette='magma',ax=ax1,legend=False)
# sns.scatterplot(data=df_2,x='Laser [um]',y='Micrometer [um]',hue='Force [lbf]',size='Run #',sizes=(50,250),palette='magma',ax=ax2,legend=False)
# sns.scatterplot(data=df_3,x='Laser [um]',y='Micrometer [um]',hue='Force [lbf]',size='Run #',sizes=(50,250),palette='magma',ax=ax3,legend=False)
# sns.scatterplot(data=df_4,x='Laser [um]',y='Micrometer [um]',hue='Force [lbf]',size='Run #',sizes=(50,250),palette='magma',ax=ax4,legend=False)
# sns.scatterplot(data=df_5,x='Laser [um]',y='Micrometer [um]',hue='Force [lbf]',size='Run #',sizes=(50,250),palette='magma',ax=ax5,legend=False)

# handles,labels = ax5.get_legend_handles_labels()
# fig.legend(handles,labels,loc='upper center')
# plt.show()


# grid = sns.FacetGrid(rawdf,col='Spring Type',hue='Force [lbf]',palette=('magma'),col_wrap=(1),margin_titles=(False),sharex=(True),sharey=(True))
# grid.map(sns.scatterplot,'Laser [um]','Micrometer [um]',legend='brief')
# # grid.set_axis_labels(y_var='')
# grid.ylabel('Test')
# grid.add_legend()



df = rawdf.filter(['Force [lbf]','Micrometer [um]','Spring Type','Run'])


g = sns.lmplot(data=df,x='Force [lbf]',y='Micrometer [um]',col='Spring Type',hue='Run')

# g = sns.relplot(data=rawdf,x='Force [lbf]',y='Micrometer [um]',col_wrap=(2),col='Spring Type',size='Run',palette='magma')
# g = sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.1}})

# plt.suptitle('Solenoid Rocker Pull Test')
# sns.set_style('whitegrid')
# sns.set_context('talk')
# ax = sns.relplot(data = rawdf,x='Laser [um]',y='Micrometer [mm]',hue='Force [lbf]',size='Run #',style='Spring Type',sizes=(50,250),palette=('magma'))
# ax.fig.set_size_inches(12,6)
# plt.show()


# sns.scatterplot(data=rawdf,x='LASER [UM]',y='MICROMETER [UM]',size='RUN',hue='FORCE [LBF]',palette='magma')
# sns.scatterplot(data=rawdf,x='Force [lbf]',y='Micrometer [um]',size='Run',palette='magma')

plt.suptitle('Top Spring Only')
# plt.legend(loc='upper left')

# plt.savefig('2022_10_25_top_spring_only.png',bbox_inches='tight')

# fig = plt.gcf()
# matplotlib.rcParams['figure.figsize'] = 14,6
# sns.set(rc={'figsize':(14,6)})
# sns.move_legend(ax,'lower center',ncol=3)
# ax.tight_layout()
# matplotlib.rcParams.update({'font.size':10})
# plt.grid()