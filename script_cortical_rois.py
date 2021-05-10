"""
This script estimates how much the given lesion affects differrnt regions 
of the Yeo atlas.

Author: Bertrand Thirion, 2021
"""
import os
from nilearn.datasets import fetch_atlas_yeo_2011
from nilearn.input_data import NiftiLabelsMasker
from nilearn.plotting import plot_roi, plot_img
from nilearn.regions import connected_label_regions
import matplotlib.pyplot as plt

data_dir = '/tmp' # directory where you put your data
lesion = os.path.join(data_dir, 'lesion.nii.gz')
# here it is assumed that your input lesion image is available at
# '/tmp/lesion.nii.gz'
# change it to any path you like

atlas = fetch_atlas_yeo_2011()
thick17 = atlas['thick_17']
# optionally display the atlas
# plot_roi(thick17)

# separate yeo17 atlas into connected components:
# 122 connected components overall
separate_thick17 = connected_label_regions(thick17)

# display the lesion on top of the atlas
display = plot_img(lesion)
display.add_overlay(thick17, cmap=plt.cm.hsv)

# compute the mean of lesion proportion per network
masker = NiftiLabelsMasker(labels_img=thick17).fit()
network_lesion = masker.transform(lesion)

# redo that on a per'region basis
display = plot_img(lesion)
display.add_overlay(separate_thick17, cmap=plt.cm.hsv)

# compute the mean of lesion proportion per network
separate_masker = NiftiLabelsMasker(labels_img=separate_thick17).fit()
region_lesion = separate_masker.transform(lesion)

plt.show()
