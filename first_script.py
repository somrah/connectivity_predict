"""
This script reads the summaries of the imaging data to create
compact features for prediction.

Author: Bertrand Thirion, 2020
"""

import pandas as pd
import numpy as np

# replace with wherever you put that file
csv_file ='/Users/lebbe/Downloads/re/all_connections.csv'
csv_file_behavior_ANTS = '/Users/lebbe/Downloads/behavior_ANTS.csv'
csv_file_behavior_FSL = '/Users/lebbe/Downloads/behavior_FSL.csv'
# this is the whole dataframe
all_connections = pd.read_csv(csv_file, index_col=0)

# get the pathway names
connection_ids = all_connections.columns[2:] # discard subjectID and TrackID
unique_ids = np.unique([c.split('_')[0] + '_'  + c.split('_')[1]
                        for c in connection_ids])

##############################################################################
# aggregate connectivity values from left, right etc.
# by summing them to a unqiue values
aggregated_connectivity = {}
for id_ in unique_ids:
    relevant_ids = [c for c in connection_ids
                    if c.startswith(id_ + '_') or c == id_]
    total_ids = [c for c in relevant_ids if c.endswith('total')]
    partial_ids = [c for c in relevant_ids if not c.endswith('total')]
    aggregated_connectivity[id_] = all_connections[partial_ids].sum(1).values
    aggregated_connectivity[id_ + '_total'] = all_connections[total_ids]\
                                              .sum(1).values

# make a dataframe from it
aggregated_connectivity = pd.DataFrame(aggregated_connectivity)
# add the missing columns
aggregated_connectivity['subjectID'] = all_connections['subjectID']

##############################################################################
# Average per subject across HCP counts
unique_subjects = all_connections['subjectID'].unique()
average_connectivity = {}
for subject in unique_subjects:
    x = aggregated_connectivity[
        aggregated_connectivity.subjectID == subject]
    average_connectivity[subject] = aggregated_connectivity[
        aggregated_connectivity.subjectID == subject].sum(0)
    # todo: more sophisticated averaging scheme to deal with low values
    # since those values are unreliable
    
# make a dataFrame from it
average_connectivity = pd.DataFrame(average_connectivity).T
average_connectivity['subjectID'] = unique_subjects

##############################################################################
# Keep onlu ANTS subjects
ANTS = [subject for subject in unique_subjects if subject.endswith('ANTS')]
FSL = [subject for subject in unique_subjects if subject.endswith('FSL')]
ANTS_connectivity = average_connectivity[
    average_connectivity.subjectID.isin(ANTS)]

# Todo: do the same with FSL_connectivity
FSL_connectivity = average_connectivity[
    average_connectivity.subjectID.isin(FSL)]
##############################################################################
# finally compute the  partial/total ratio in each subject 
ANTS_ratio = {}
for id_ in unique_ids:
    ANTS_ratio[id_] = ANTS_connectivity[id_] / (
        1. + ANTS_connectivity[id_ + '_total'])  

# make a DataFrame from it
ANTS_ratio = pd.DataFrame(ANTS_ratio)
# ANTS_ratio supposeldy contains some data that are ready for machine learning
#do the same with FSL_connectivity
FSL_ratio={}
for id_ in unique_ids:
    FSL_ratio[id_] = FSL_connectivity[id_] / (
    1. + FSL_connectivity[id_+'_total'])

#make a DataFrame from it : 
FSL_ratio = pd.DataFrame(FSL_ratio)

##############################################################################
#plot ANTS ratio (corticocortical, corticostriatal, corticothalamic)
import matplotlib.pyplot as plt
from math import pi
#make radar plot corticocortical networks for each subject
#define number of variables
def make_spider (row, title, color):
    categories=list(ANTS_ratio) [0:17]
    N=len(categories)
#define the angle for each variable
    angles=[n/float(N)*2*pi for n in range(N)]
    angles += angles[:1]
#initialize the spider plot
    ax = plt.subplot(2,2,row+1, polar=True, )
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [i.strip('CorticoCortical_') for i in categories], color='black', size=6)
     
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1,0.3,0.5,0.7,0.9], ["0.1","0.3","0.5","0.7","0.9"], color="grey", size=7)
    plt.ylim(0,0.2)
     
    # Ind1
    values=ANTS_ratio.loc[row].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values[0:17], color='grey', linewidth=2, linestyle='solid')
    ax.fill(angles, values [0:17], color='grey', alpha=0.4)
     
# Add a title
    plt.title('corticocortical', size=11, color='grey', y=1.1 ) 
          # ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(ANTS_ratio.index))
 
# Loop to plot
for row in range(0, len(ANTS_ratio.index)):
    make_spider (row=row, title=ANTS_ratio.index[row], color=my_palette(row))

#behavior_ANTS = pd.read_csv(csv_file_behavior_ANTS, index_col=0)
#add PIMM (perseveration early postop : 1 indicates the presence of perseveration)
#ANTS_ratio['PIMM']=behavior_ANTS['PIMM']
#add 3M (perseveration latepostop : 1 indicates the presence of perseveration)
#ANTS_ratio['3M']=behavior_ANTS['3M']
#add boucle (severe perseveration during surgery : 1 indicates the presence of perseveration
#ANTS_ratio['BOUCLE']=behavior_ANTS['PEROPBOUCLE']
#plot ratios for subjects with presence of perseveration early postop vs others
#groups = ANTS_ratio.groupby('PIMM')


#FSL_ratio.index
#behavior_FSL=behavior
#behavior_FSL(index)
#ANTS_ratio.index
