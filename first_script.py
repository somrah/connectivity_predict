"""
This script reads the summaries of the imaging data to create
compact features for prediction.

Author: Bertrand Thirion, 2020
"""

import pandas as pd
import numpy as np

# replace with wherever you put that file
csv_file ='/Users/lebbe/Downloads/re/all_connections.csv'
csv_file_behavior = '/Users/lebbe/Downloads/behavior.csv'

#directory where figures are saved
mypath = '/Users/lebbe/Downloads/'

# this is the whole dataframe
all_connections = pd.read_csv(csv_file, index_col=0)
#this is behavioral data ( 1 indicates the presence of perseveration)
behavior = pd.read_csv(csv_file_behavior, index_col=0)
# get the pathway names
connection_ids = all_connections.columns[2:] # discard subjectID and TrackID
unique_ids = np.unique([c.split('_')[0] + '_'  + c.split('_')[1]
                        for c in connection_ids])

##############################################################################
# aggregate connectivity values from left, right etc.
# by summing them to a unique values
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

#add subject ID
average_connectivity['subjectID'] = unique_subjects
##############################################################################
#other methods to deal with low values : 
# method BIS : sum of ratios
# 1) for each subject make a sum of ratios obtained in each HCP connectome 
# 2) transform ratios with sigmoid function
unique_subjects = all_connections['subjectID'].unique()
ratios_connectivity_bis_HCP = {}
sumratios_connectivity_bis_subject = {}

for id_ in unique_ids:
    ratios_connectivity_bis_HCP[id_]= aggregated_connectivity[id_]/(
        1.+ aggregated_connectivity[id_ + '_total'])
ratios_connectivity_bis_HCP = pd.DataFrame(ratios_connectivity_bis_HCP)
ratios_connectivity_bis_HCP['subjectID']=all_connections['subjectID']

for subject in unique_subjects:
    x=ratios_connectivity_bis_HCP[
            ratios_connectivity_bis_HCP.subjectID == subject]
    sumratios_connectivity_bis_subject [subject] = ratios_connectivity_bis_HCP[
            ratios_connectivity_bis_HCP.subjectID == subject].sum(0)
sumratios_connectivity_bis_subject = pd.DataFrame(sumratios_connectivity_bis_subject).T
sumratios_connectivity_bis_subject_transformed= 1/(1+np.exp(np.asarray(-sumratios_connectivity_bis_subject.drop(['subjectID'], axis = 1),dtype=float)))
sumratios_connectivity_bis_subject_transformed=pd.DataFrame(sumratios_connectivity_bis_subject_transformed)
sumratios_connectivity_bis_subject_transformed['subjectID'] = unique_subjects
ANTS = [subject for subject in unique_subjects if subject.endswith('ANTS')]
ANTS_sumratios_connectivity_bis_subject_transformed=sumratios_connectivity_bis_subject_transformed[sumratios_connectivity_bis_subject_transformed.subjectID.isin(ANTS)]
##############################################################################
# Keep only ANTS subjects
ANTS = [subject for subject in unique_subjects if subject.endswith('ANTS')]

ANTS_connectivity = average_connectivity[
    average_connectivity.subjectID.isin(ANTS)]

# Todo: do the same with FSL_connectivity
FSL = [subject for subject in unique_subjects if subject.endswith('FSL')]
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
#transform data with sigmoid function 
ANTS_ratio_transformed= 1/(1+np.exp(np.asarray(-ANTS_ratio,dtype=float)))
ANTS_ratio_transformed=pd.DataFrame(ANTS_ratio_transformed)
# ANTS_ratio supposeldy contains some data that are ready for machine learning
#do the same with FSL_connectivity
FSL_ratio={}
for id_ in unique_ids:
    FSL_ratio[id_] = FSL_connectivity[id_] / (
    1. + FSL_connectivity[id_+'_total'])

#make a DataFrame from it : 
FSL_ratio = pd.DataFrame(FSL_ratio)
#transform data with sigmoid function 
FSL_ratio_transformed= 1/(1+np.exp(np.asarray(-FSL_ratio,dtype=float)))
FSL_ratio_transformed=pd.DataFrame(FSL_ratio_transformed)
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
    ax = plt.subplot(4,10,row+1, polar=True, )
    #Add behavior
    #ax.text(3, 8, 'PV early postop =' + str(behavior.values[row,3]), fontsize=14), style='italic',
     #   bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(0.95, 0.06, 'PV early postop =' + str(behavior.values[row,3]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=12) 
    
   # ax.annotate('PV late postop =' + str(behavior[row,0]), fontsize=14, xy=(2, 1), xytext=(3, 4),
   #arrowprops=dict(facecolor='red', shrink=0.05))
    ax.text(0.95, 0.005, 'PV late postop =' + str(behavior.values[row,0]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=12) 
    ax.text(0.95, 0.11, 'PV boucle perop =' + str(behavior.values[row,2]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='purple', fontsize=12) 
    #ax.text(0, -1, 'PV early postop =' + str(behavior.values[row,3]), fontsize=15)
    #plt.gcf().text(0.8, 0.8, 'PV early postop =' + str(behavior.values[row,3]), fontsize=14)
    #plt.gcf().text(0.7, 0.8, 'PV late postop =' + str(behavior[row,0]), fontsize=14)
    #plt.gcf().text(0.6, 0.8, 'PV boucle perop =' + str(behavior[row,2]), fontsize=14)
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [i.strip('CorticoCortical_') for i in categories], color='black', size=6)
    #plt.xticks(angles[:-1], categories, color='black', size=6)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1,0.3,0.5,0.7,0.9], ["0.1","0.3","0.5","0.7","0.9"], color="grey", size=7)
    plt.ylim(0,1)
     
    # Ind1
    values=ANTS_ratio.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0:17], values[0:17], color=color, linewidth=2, linestyle='solid')
    ax.fill(angles[0:17], values [0:17], color=color, alpha=0.4)
     
# Add a title
    plt.title(title, size=11, color='grey', y=1.1 )

    
          # ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(4000/my_dpi, 4000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(ANTS_ratio.index))
 
# Give a name to big figure
plt.gcf().text(0.9, 0.9, 'corticocortical_ANTS', fontsize=40)
# Loop to plot
for row in range(0, len(ANTS_ratio.index)):
    make_spider (row=row, title=ANTS_ratio.index[row], color=my_palette(row))
#save figure 
plt.savefig(mypath + 'corticocortical_ANTS.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#make radar plot corticoSTRIATAL (available in dataset n=15) networks for each subject
#define number of variables
def make_spider (row, title, color):
    categories=list(ANTS_ratio) [17:32]
    N=len(categories)
    #define the angle for each variable
    angles=[n/float(N)*2*pi for n in range(N)]
    angles += angles[:1]
    
    #initialize the spider plot
    ax = plt.subplot(4,10,row+1, polar=True, )
    #Add behavior
    #ax.text(3, 8, 'PV early postop =' + str(behavior.values[row,3]), fontsize=14), style='italic',
     #   bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(0.95, 0.06, 'PV early postop =' + str(behavior.values[row,3]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=12) 
    
   # ax.annotate('PV late postop =' + str(behavior[row,0]), fontsize=14, xy=(2, 1), xytext=(3, 4),
   #arrowprops=dict(facecolor='red', shrink=0.05))
    ax.text(0.95, 0.005, 'PV late postop =' + str(behavior.values[row,0]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=12) 
    ax.text(0.95, 0.11, 'PV boucle perop =' + str(behavior.values[row,2]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='purple', fontsize=12) 
    #ax.text(0, -1, 'PV early postop =' + str(behavior.values[row,3]), fontsize=15)
    #plt.gcf().text(0.8, 0.8, 'PV early postop =' + str(behavior.values[row,3]), fontsize=14)
    #plt.gcf().text(0.7, 0.8, 'PV late postop =' + str(behavior[row,0]), fontsize=14)
    #plt.gcf().text(0.6, 0.8, 'PV boucle perop =' + str(behavior[row,2]), fontsize=14)
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [i.strip('CorticoStriatal_') for i in categories], color='black', size=6)
    #plt.xticks(angles[:-1], categories, color='black', size=6)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1,0.3,0.5,0.7,0.9], ["0.1","0.3","0.5","0.7","0.9"], color="grey", size=7)
    plt.ylim(0,1)
     
    # Ind1
    values=ANTS_ratio.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0:15], values[17:32], color=color, linewidth=2, linestyle='solid')
    ax.fill(angles[0:15], values [17:32], color=color, alpha=0.4)
     
# Add a title
    plt.title(title, size=11, color='grey', y=1.1 )

    
          # ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(4000/my_dpi, 4000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set3", len(ANTS_ratio.index))
 
# Give a name to big figure
plt.gcf().text(0.9, 0.9, 'corticostriatal_ANTS', fontsize=40)
# Loop to plot
for row in range(0, len(ANTS_ratio.index)):
    make_spider (row=row, title=ANTS_ratio.index[row], color=my_palette(row))
#save figure 
plt.savefig(mypath+'corticostriatal_ANTS.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
#make radar plot corticoTHALAMIC networks (available in dataset n=14) for each subject
#define number of variables
def make_spider (row, title, color):
    categories=list(ANTS_ratio) [32:46]
    N=len(categories)
    #define the angle for each variable
    angles=[n/float(N)*2*pi for n in range(N)]
    angles += angles[:1]
    
    #initialize the spider plot
    ax = plt.subplot(4,10,row+1, polar=True, )
    #Add behavior
    #ax.text(3, 8, 'PV early postop =' + str(behavior.values[row,3]), fontsize=14), style='italic',
     #   bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(0.95, 0.06, 'PV early postop =' + str(behavior.values[row,3]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=12) 
    
   # ax.annotate('PV late postop =' + str(behavior[row,0]), fontsize=14, xy=(2, 1), xytext=(3, 4),
   #arrowprops=dict(facecolor='red', shrink=0.05))
    ax.text(0.95, 0.005, 'PV late postop =' + str(behavior.values[row,0]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=12) 
    ax.text(0.95, 0.11, 'PV boucle perop =' + str(behavior.values[row,2]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='purple', fontsize=12) 
  
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [i.strip('CorticoThalamic_') for i in categories], color='black', size=6)
    #plt.xticks(angles[:-1], categories, color='black', size=6)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1,0.3,0.5,0.7,0.9], ["0.1","0.3","0.5","0.7","0.9"], color="grey", size=7)
    plt.ylim(0,1)
     
    # Ind1
    values=ANTS_ratio.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0:14], values[32:46], color=color, linewidth=2, linestyle='solid')
    ax.fill(angles[0:14], values [32:46], color=color, alpha=0.4)
     
# Add a title
    plt.title(title, size=11, color='grey', y=1.1 )

    
          # ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(4000/my_dpi, 4000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set1", len(ANTS_ratio.index))
 
# Give a name to big figure
plt.gcf().text(0.9, 0.9, 'corticothalamic_ANTS', fontsize=40)
# Loop to plot
for row in range(0, len(ANTS_ratio.index)):
    make_spider (row=row, title=ANTS_ratio.index[row], color=my_palette(row))
#save figure 
plt.savefig(mypath + 'corticothalamic_ANTS.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#plot FSL ratio (corticocortical, corticostriatal, corticothalamic)
import matplotlib.pyplot as plt
from math import pi
#make radar plot corticocortical networks for each subject
#define number of variables
def make_spider (row, title, color):
    categories=list(FSL_ratio) [0:17]
    N=len(categories)
    #define the angle for each variable
    angles=[n/float(N)*2*pi for n in range(N)]
    angles += angles[:1]
    
    #initialize the spider plot
    ax = plt.subplot(4,10,row+1, polar=True, )
    #Add behavior
    #ax.text(3, 8, 'PV early postop =' + str(behavior.values[row,3]), fontsize=14), style='italic',
     #   bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(0.95, 0.06, 'PV early postop =' + str(behavior.values[row,3]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=12) 
    
   # ax.annotate('PV late postop =' + str(behavior[row,0]), fontsize=14, xy=(2, 1), xytext=(3, 4),
   #arrowprops=dict(facecolor='red', shrink=0.05))
    ax.text(0.95, 0.005, 'PV late postop =' + str(behavior.values[row,0]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=12) 
    ax.text(0.95, 0.11, 'PV boucle perop =' + str(behavior.values[row,2]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='purple', fontsize=12) 

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [i.strip('CorticoCortical_') for i in categories], color='black', size=6)
    #plt.xticks(angles[:-1], categories, color='black', size=6)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1,0.3,0.5,0.7,0.9], ["0.1","0.3","0.5","0.7","0.9"], color="grey", size=7)
    plt.ylim(0,1)
     
    # Ind1
    values=FSL_ratio.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0:17], values[0:17], color=color, linewidth=2, linestyle='solid')
    ax.fill(angles[0:17], values [0:17], color=color, alpha=0.4)
     
# Add a title
    plt.title(title, size=11, color='grey', y=1.1 )

    
          # ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(4000/my_dpi, 4000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set2", len(ANTS_ratio.index))
 
# Give a name to big figure
plt.gcf().text(0.9, 0.9, 'corticocortical_FSL', fontsize=40)
# Loop to plot
for row in range(0, len(FSL_ratio.index)):
    make_spider (row=row, title=FSL_ratio.index[row], color=my_palette(row))
#save figure 
plt.savefig(mypath +'corticocortical_FSL.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#make radar plot corticoSTRIATAL (available in dataset n=15) networks for each subject
#define number of variables
def make_spider (row, title, color):
    categories=list(FSL_ratio) [17:32]
    N=len(categories)
    #define the angle for each variable
    angles=[n/float(N)*2*pi for n in range(N)]
    angles += angles[:1]
    
    #initialize the spider plot
    ax = plt.subplot(4,10,row+1, polar=True, )
    #Add behavior
    #ax.text(3, 8, 'PV early postop =' + str(behavior.values[row,3]), fontsize=14), style='italic',
     #   bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(0.95, 0.06, 'PV early postop =' + str(behavior.values[row,3]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=12) 
    
   # ax.annotate('PV late postop =' + str(behavior[row,0]), fontsize=14, xy=(2, 1), xytext=(3, 4),
   #arrowprops=dict(facecolor='red', shrink=0.05))
    ax.text(0.95, 0.005, 'PV late postop =' + str(behavior.values[row,0]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=12) 
    ax.text(0.95, 0.11, 'PV boucle perop =' + str(behavior.values[row,2]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='purple', fontsize=12) 

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [i.strip('CorticoStriatal_') for i in categories], color='black', size=6)
    #plt.xticks(angles[:-1], categories, color='black', size=6)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1,0.3,0.5,0.7,0.9], ["0.1","0.3","0.5","0.7","0.9"], color="grey", size=7)
    plt.ylim(0,1)
     
    # Ind1
    values=FSL_ratio.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0:15], values[17:32], color=color, linewidth=2, linestyle='solid')
    ax.fill(angles[0:15], values [17:32], color=color, alpha=0.4)
     
# Add a title
    plt.title(title, size=11, color='grey', y=1.1 )

    
          # ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(4000/my_dpi, 4000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set3", len(FSL_ratio.index))
 
# Give a name to big figure
plt.gcf().text(0.9, 0.9, 'corticostriatal_FSL', fontsize=40)
# Loop to plot
for row in range(0, len(FSL_ratio.index)):
    make_spider (row=row, title=FSL_ratio.index[row], color=my_palette(row))
#save figure 
plt.savefig(mypath +'corticostriatal_FSL.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)
#make radar plot corticoTHALAMIC networks (available in dataset n=14) for each subject
#define number of variables
def make_spider (row, title, color):
    categories=list(FSL_ratio) [32:46]
    N=len(categories)
    #define the angle for each variable
    angles=[n/float(N)*2*pi for n in range(N)]
    angles += angles[:1]
    
    #initialize the spider plot
    ax = plt.subplot(4,10,row+1, polar=True, )
    #Add behavior
    #ax.text(3, 8, 'PV early postop =' + str(behavior.values[row,3]), fontsize=14), style='italic',
     #   bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    ax.text(0.95, 0.06, 'PV early postop =' + str(behavior.values[row,3]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='blue', fontsize=12) 
    
   # ax.annotate('PV late postop =' + str(behavior[row,0]), fontsize=14, xy=(2, 1), xytext=(3, 4),
   #arrowprops=dict(facecolor='red', shrink=0.05))
    ax.text(0.95, 0.005, 'PV late postop =' + str(behavior.values[row,0]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='red', fontsize=12) 
    ax.text(0.95, 0.11, 'PV boucle perop =' + str(behavior.values[row,2]),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='purple', fontsize=12) 

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], [i.strip('CorticoThalamic_') for i in categories], color='black', size=6)
    #plt.xticks(angles[:-1], categories, color='black', size=6)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1,0.3,0.5,0.7,0.9], ["0.1","0.3","0.5","0.7","0.9"], color="grey", size=7)
    plt.ylim(0,1)
     
    # Ind1
    values=FSL_ratio.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0:14], values[32:46], color=color, linewidth=2, linestyle='solid')
    ax.fill(angles[0:14], values [32:46], color=color, alpha=0.4)
     
# Add a title
    plt.title(title, size=11, color='grey', y=1.1 )

    
          # ------- PART 2: Apply to all individuals
# initialize the figure
my_dpi=96
plt.figure(figsize=(4000/my_dpi, 4000/my_dpi), dpi=my_dpi)
 
# Create a color palette:
my_palette = plt.cm.get_cmap("Set1", len(FSL_ratio.index))
 
# Give a name to big figure
plt.gcf().text(0.9, 0.9, 'corticothalamic_FSL', fontsize=40)
# Loop to plot
for row in range(0, len(FSL_ratio.index)):
    make_spider (row=row, title=FSL_ratio.index[row], color=my_palette(row))
#save figure 
plt.savefig(mypath +'corticothalamic_FSL.png', dpi=None, facecolor='w', edgecolor='w',
        orientation='landscape', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

##############################################################################
# predict presence of verbal perseveration by applying random forrest analysis
from sklearn.model_selection import (ShuffleSplit, cross_val_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import (IncrementalPCA, SparsePCA)
#verbal perseverations early postop

#first method

X=ANTS_ratio.iloc[:, 0:46]
Y=behavior.iloc[:,3]

#remove subject with nan value from both datasets (here the second line)
X=X.drop(X.index[1])
Y=Y.drop(Y.index[1])

#we need to perform cross-validation with lots of folds 
rs = ShuffleSplit(n_splits=100, test_size=.2, random_state=0)
clf=RandomForestClassifier (n_estimators=45, random_state=0) 
scores = cross_val_score(clf, X,Y, cv=rs)
# print accuracy
print ("Accuracy first method : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))

#first method with transformation

X=ANTS_ratio_transformed.iloc[:, 0:46]
Y=behavior.iloc[:,3]

#remove subject with nan value from both datasets (here the second line)
X=X.drop(X.index[1])
Y=Y.drop(Y.index[1])

#we need to perform cross-validation with lots of folds 
rs = ShuffleSplit(n_splits=100, test_size=.2, random_state=0)
clf=RandomForestClassifier (n_estimators=45, random_state=0) 
scores = cross_val_score(clf, X,Y, cv=rs)
# print accuracy
print ("Accuracy first method with transformation : %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))

#method bis with transformation
X=ANTS_sumratios_connectivity_bis_subject_transformed.iloc[:, 0:46]
Y=behavior.iloc[:,3]

#remove subject with nan value from both datasets (here the second line)
X=X.drop(X.index[1])
Y=Y.drop(Y.index[1])

#we need to perform cross-validation with lots of folds 
rs = ShuffleSplit(n_splits=100, test_size=.2, random_state=0)
clf=RandomForestClassifier (n_estimators=45, random_state=0) 
scores = cross_val_score(clf, X,Y, cv=rs)
# print accuracy
print ("Accuracy method bis with transformation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))

#first method with iPCA

X=ANTS_ratio.iloc[:, 0:46]
Y=behavior.iloc[:,3]

#remove subject with nan value from both datasets (here the second line)
X=X.drop(X.index[1])
Y=Y.drop(Y.index[1])
#apply iPCA to data
n_components = None
ipca = IncrementalPCA(n_components=n_components, batch_size=None)
X_ipca=ipca.fit_transform(X)
#we need to perform cross-validation with lots of folds 
rs = ShuffleSplit(n_splits=100, test_size=.2, random_state=0)
clf=RandomForestClassifier (n_estimators=45, random_state=0) 
scores = cross_val_score(clf, X_ipca,Y, cv=rs)
# print accuracy
print ("Accuracy first method with iPCA: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
 
#first method with sparse PCA

X=ANTS_ratio.iloc[:, 0:46]
Y=behavior.iloc[:,3]

#remove subject with nan value from both datasets (here the second line)
X=X.drop(X.index[1])
Y=Y.drop(Y.index[1])
#apply sPCA to data


transformer = SparsePCA(n_components=5, random_state=0)
transformer.fit(X)
SparsePCA(...)
X_transformed = transformer.transform(X)
#we need to perform cross-validation with lots of folds 
rs = ShuffleSplit(n_splits=100, test_size=.2, random_state=0)
clf=RandomForestClassifier (n_estimators=45, random_state=0) 
scores = cross_val_score(clf, X_transformed,Y, cv=rs)
# print accuracy
print ("Accuracy first method with sPCA: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
 

