# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 02:21:33 2023

@author: Heitor Nunes Rosa
@gmail: heitornunes12@gmail.com
@github: @hnrosa

"""

# %%

import matplotlib.pyplot as plt
import seaborn as sns
from palmerpenguins import load_penguins

data = load_penguins()

# %%

fig, ax = plt.subplots(1, 2, figsize=(13, 6))

sns.scatterplot(data=data, x='bill_length_mm', y='bill_depth_mm', ax=ax[0])

sns.scatterplot(data=data, x='bill_length_mm',
                y='bill_depth_mm',
                hue='species',
                style='sex',
                size='body_mass_g',
                sizes=(10, 150),
                markers=('s', '^'),
                palette='inferno', ax=ax[1])

plt.legend(bbox_to_anchor=(1, 1))

plt.savefig('five_info_plot.png', dpi=500)
