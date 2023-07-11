# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 22:27:03 2023

@author: Heitor Nunes Rosa
@gmail: heitornunes12@gmail.com
"""

import matplotlib.pyplot as plt
import seaborn as sns

# %%

def completedistplot(df, xx, color = 'b',  background = 'light',
                     hist = True, kde = True, strip = True, rug = True, box = True,
                     hist_kws = None, kde_kws = None, strip_kws = None, 
                     rug_kws = None, box_kws = None, fig = None, ax1 = None):
    
    custom_kws = (hist_kws, kde_kws, strip_kws, rug_kws, box_kws)
    
    # Select edge colors based on the background
    if background == 'light':
        linecolor = 'k'
    else:
        linecolor = 'w'
        
    
    # Setting standard configurations        
    hist_std_kws = {'stat':'density', 'fill':False, 'color': linecolor}
    
    kde_std_kws = {'linewidth':2, 'color': color}
    
    strip_std_kws = {'color': color, 'alpha':0.4, 'size':8}
    
    rug_std_kws = {'color': color, 'height': 0.05}
    
    box_std_kws = {'meanline': True,
    'showmeans': True,
    'linewidth': 2, 
    'boxprops' : {'edgecolor': linecolor},
    'medianprops' : {'color': linecolor},
    'meanprops': {'linestyle': '--', 'linewidth': 3, 'color': color},
    'whiskerprops': {'color': linecolor},
    'capprops': {'color': linecolor},
    'alpha': 0.,
    }
    
    standard_kws = (hist_std_kws, kde_std_kws, strip_std_kws, rug_std_kws,
               box_std_kws)
    
    
    # Changing standard configurations if is there any custom configuration
    for cus_kws, std_kws in zip(custom_kws, standard_kws):
        
        if cus_kws is not None:
            for k, v in cus_kws.items():
                std_kws[k] = v
                
                
    # Creating figure if None is provided           
    if fig is None and ax1 is None:
        fig, ax1 = plt.subplots(2, 1, figsize =  (12, 6))
        
        
    # Plotting desired distributions
    if hist:
        sns.histplot(data = data, x = xx, ax = ax1[0], **hist_std_kws)
    
    if kde:
        sns.kdeplot(data = data, x = xx, ax = ax1[0], **kde_std_kws)
    
    if strip:
        sns.stripplot(data = data, x = xx, ax = ax1[1], **strip_std_kws)
    
    if rug:
        sns.rugplot(data = data, x = xx, ax = ax1[1], **rug_std_kws)
    
    if box:
        alpha = box_std_kws['alpha']
        
        box_std_kws.pop('alpha')
        
        ax_ = sns.boxplot(data = data, x = xx, ax = ax1[1], **box_std_kws)
        
    # Setting alpha configuration for boxplot
        for patch in ax_.patches:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, alpha))
            
    # Removing features from first axis
    ax1[0].set_xlabel('')
    ax1[0].set_ylabel('')

    ax1[0].tick_params(
                   bottom=False,
                   left=False,
                   labelleft=False,
                   labelbottom=False)

    ax1[0].spines['top'].set_visible(False)
    ax1[0].spines['left'].set_visible(False)
    ax1[0].spines['right'].set_visible(False)

    # Setting tick and xlabel configurations for second axis
    ax1[1].tick_params(labelsize = 20)
    ax1[1].set_xlabel(xx, size = 20)

    # Removing features from second axis
    ax1[1].spines['top'].set_visible(False)
    ax1[1].spines['left'].set_visible(False)
    ax1[1].spines['right'].set_visible(False)

    
    # Setting limits for both axis
    ax1[0].set_xlim(0.97 * data[xx].min(), data[xx].max() * 1.03)
    ax1[1].set_xlim(0.97 * data[xx].min(), data[xx].max() * 1.03)
    
    fig.tight_layout()
    
    return fig, ax1


# %%

if __name__ == "__main__":
    
    from palmerpenguins import load_penguins
    
    data = load_penguins()
    
    plt.style.use('default')
    _, _ = completedistplot(data, 'bill_length_mm');
    
    _, _ = completedistplot(data, 'bill_length_mm', color = 'r',
                               hist_kws= {'bins': 20}, 
                               kde_kws = {'linewidth':5});
    
    _, _ = completedistplot(data, 'bill_length_mm', color = 'g',
                               hist_kws= {'bins': 20}, rug = False);
    
    _, _ = completedistplot(data, 'bill_length_mm', color = 'b',
                               hist_kws = {'bins': 20, 'fill': True, 
                                           'color': '#fcdb03', 'alpha': 0.6},
                               box_kws = {'color': '#fcdb03', 'alpha': 0.6});
    
    plt.style.use('dark_background')
    _, _ = completedistplot(data, 'bill_length_mm', color = 'b', 
                            background = 'dark',
                               hist_kws = {'bins': 20, 'fill': True, 
                                           'color': '#fcdb03', 'alpha': 0.6},
                               box_kws = {'color': '#fcdb03', 'alpha': 0.6});



    
            
    
        
    



