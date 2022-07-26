#%%
import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns

def list_files(filepath, filetype):
   paths = []
   for root, dirs, files in os.walk(filepath):
      for file in files:
         if file.lower().endswith(filetype.lower()):
            paths.append(os.path.join(root, file))
   return(paths)
# %%
path = "artifacts/corpus"
settings_list = list_files(path, 'heldout.npy')

K = []
values = []
gamma = []
model_type = []
for setting in settings_list: 
    # grab values
    heldout = np.load(setting)
    k_value = setting.split('_')[1]
    gamma_factor = setting.split('/')[2].split('_')[-1]
    model = setting.split('/')[-2]
    # append each model run settings
    gamma.append(gamma_factor)
    model_type.append(model)
    K.append(k_value)
    values.append(heldout)

# create dataframe
results = pd.DataFrame(
    {
    'model':model_type,
    'gamma_factor':np.int0(gamma),
    'K': K,
    'heldout': np.round(values,4),
    })
#%% 0 vs.1 

data = results[(results.gamma_factor == 0) | (results.gamma_factor == 1)]
p = sns.catplot(
   x="K",
   y="heldout",
   hue="model",
   col = 'gamma_factor',
   data=data,
   kind='box',
   sharey=False,
   ci='sd',
   order = ['10','30','50','70'])

p.set_axis_labels('Number of Topics', 'Per Word Heldout Likelihood')
p.set_titles('')

hatches = ['', '//']
# iterate through each subplot / Facet
for ax in p.axes.flat:
   ax.spines['left'].set_color('k')
   ax.xaxis.label.set_color('k')
   ax.yaxis.label.set_color('k')
   # select the correct patches
   patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
   # the number of patches should be evenly divisible by the number of hatches
   h = hatches * (len(patches) // len(hatches))
   # iterate through the patches for each subplot
   for patch, hatch in zip(patches, h):
      patch.set_hatch(hatch)
      patch.set_edgecolor('k')
      for lp, hatch in zip(p.legend.get_patches(), hatches):
         lp.set_hatch(hatch)
plt.savefig('img/0_1_bp.png', bbox_inches='tight', dpi = 400)
#%% 1 vs. 3
data = results[(results.gamma_factor == 1) | (results.gamma_factor == 3)]
p = sns.catplot(
   x="K",
   y="heldout",
   hue="model",
   col = 'gamma_factor',
   data=data,
   kind='box',
   sharey=False,
   ci='sd',
   order = ['10','30','50','70'])

p.set_axis_labels('Number of Topics', 'Per Word Heldout Likelihood')
p.set_titles('')

hatches = ['', '//']
# iterate through each subplot / Facet
for ax in p.axes.flat:
   ax.spines['left'].set_color('k')
   ax.xaxis.label.set_color('k')
   ax.yaxis.label.set_color('k')
   # select the correct patches
   patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
   # the number of patches should be evenly divisible by the number of hatches
   h = hatches * (len(patches) // len(hatches))
   # iterate through the patches for each subplot
   for patch, hatch in zip(patches, h):
      patch.set_hatch(hatch)
      patch.set_edgecolor('k')
      for lp, hatch in zip(p.legend.get_patches(), hatches):
         lp.set_hatch(hatch)

plt.savefig('img/1_3_bp.png', bbox_inches='tight', dpi = 400)
#%% 3 vs. 5

data = results[(results.gamma_factor == 3) | (results.gamma_factor == 5)]
p = sns.catplot(
   x="K",
   y="heldout",
   hue="model",
   col ="gamma_factor",
   data=data,
   kind="box",
   dodge=True,
   ci="sd",
   sharey=False,
   order = ['10','30','50','70'],
   legend_out=False,
   )

p.set_axis_labels('Number of Topics', 'Per Word Heldout Likelihood')
p.set_titles('')

hatches = ['', '//']
# iterate through each subplot / Facet
for ax in p.axes.flat:
   #ax.grid()
   ax.spines['left'].set_color('k')
   ax.xaxis.label.set_color('k')
   ax.yaxis.label.set_color('k')

   # select the correct patches
   patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
   # the number of patches should be evenly divisible by the number of hatches
   h = hatches * (len(patches) // len(hatches))
   # iterate through the patches for each subplot
   for patch, hatch in zip(patches, h):
      patch.set_hatch(hatch)
      patch.set_edgecolor('k')
      for lp, hatch in zip(p.legend.get_patches(), hatches):
         lp.set_hatch(hatch)

plt.savefig('img/3_5_bp.png', bbox_inches='tight', dpi = 400)
#%% 5 vs. 10 

data = results[(results.gamma_factor == 5) | (results.gamma_factor == 10)]
p = sns.catplot(
   x="K",
   y="heldout",
   hue="model",
   col ="gamma_factor",
   data=data,
   kind="box",
   dodge=True,
   ci="sd",
   sharey=False,
   order = ['10','30','50','70'],
   legend_out=False,
   )

p.set_axis_labels('Number of Topics', 'Per Word Heldout Likelihood')
p.set_titles('')

hatches = ['', '//']
# iterate through each subplot / Facet
for ax in p.axes.flat:
   #ax.grid()
   ax.spines['left'].set_color('k')
   ax.xaxis.label.set_color('k')
   ax.yaxis.label.set_color('k')

   # select the correct patches
   patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]
   # the number of patches should be evenly divisible by the number of hatches
   h = hatches * (len(patches) // len(hatches))
   # iterate through the patches for each subplot
   for patch, hatch in zip(patches, h):
      patch.set_hatch(hatch)
      patch.set_edgecolor('k')
      for lp, hatch in zip(p.legend.get_patches(), hatches):
         lp.set_hatch(hatch)
plt.savefig('img/5_10_bp.png', bbox_inches='tight', dpi = 400)
# %% descriptives
results.loc[:,['K','heldout']].groupby(['K']).mean()
# %%
results.loc[:,['model', 'K', 'heldout','gamma_factor']].groupby(['K','model','gamma_factor']).median()

# %%
