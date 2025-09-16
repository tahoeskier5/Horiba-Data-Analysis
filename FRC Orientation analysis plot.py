import pandas as pd
import matplotlib.pyplot as plt

# 1. Define your raw data
channel_ids = [1]*4 + [2]*4 + [3]*4 + [4]*4
setpoints   = [250, 500, 750, 1000] * 4

ori2_diff = [
    0.198564719,  0.091342199,  0.029061404, -0.014211383,
    0.190195672,  0.056289658, -0.020201025, -0.053908351,
    0.192676645,  0.043672413, -0.001599580, -0.049907213,
    0.276810183,  0.156015042,  0.105754349,  0.086082458
]
ori3_diff = [
   -0.316750603, -0.225051352, -0.202184924, -0.179553638,
   -0.490026013, -0.392144460, -0.364904823, -0.338650821,
   -0.541642752, -0.482247793, -0.446050180, -0.435863422,
   -0.524304273, -0.494363750, -0.469857314, -0.448919346
]
ori4_diff = [
   -0.646800305, -0.532567836, -0.490761261, -0.453852476,
   -0.702345261, -0.614511738, -0.578405488, -0.543065062,
   -0.752266886, -0.645922191, -0.586751103, -0.566156795,
   -0.584944254, -0.482242123, -0.428923830, -0.388632736
]
ori5_diff = [
   -0.640217404, -0.479571726, -0.440767268, -0.402790968,
   -0.690836655, -0.612611666, -0.589982427, -0.561903602,
   -0.716230755, -0.619918057, -0.564233177, -0.551196950,
   -0.642533417, -0.507916804, -0.446994295, -0.403770722
]

# 2. Build a DataFrame in long form
df = pd.DataFrame({
    'Channel ID':     channel_ids,
    'Setpoint':       setpoints,
    'Orientation 2':  ori2_diff,
    'Orientation 3':  ori3_diff,
    'Orientation 4':  ori4_diff,
    'Orientation 5':  ori5_diff,
})

df_long = df.melt(
    id_vars=['Channel ID', 'Setpoint'],
    value_vars=['Orientation 2','Orientation 3','Orientation 4','Orientation 5'],
    var_name='Orientation',
    value_name='Flow Accuracy Difference [%RDG]'
)

# 3. Prepare data for boxplot: group by Orientation
groups = [
    df_long.loc[df_long['Orientation']==ori, 'Flow Accuracy Difference [%RDG]']
    for ori in ['Orientation 2','Orientation 3','Orientation 4','Orientation 5']
]

plt.rcParams.update({
    'font.size': 14,           # default text size
    'axes.titlesize': 18,      # title size
    'axes.labelsize': 16,      # axis label size
    'xtick.labelsize': 14,     # x-tick label size
    'ytick.labelsize': 14      # y-tick label size
})


# 4. Create the boxplot
fig, ax = plt.subplots(figsize=(8,6))
ax.boxplot(groups, labels=['Orientation 2','Orientation 3','Orientation 4','Orientation 5'])

ax.set_xlabel('Orientation')
ax.set_ylabel('Flow Accuracy Difference [%RDG]')
ax.set_title('Distribution of Flow Accuracy Difference by Orientation')
ax.grid(True)
plt.tight_layout()
plt.show()
