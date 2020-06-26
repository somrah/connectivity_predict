"""
This script reads the summaries of the imaging data to create
compact features for prediction.

Author: Bertrand Thirion, 2020
"""

import pandas as pd
import numpy as np

# replace with wherever you put that file
csv_file = '/tmp/all_connections.csv'

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

##############################################################################
# finally compute the  partial/total ratio in each subject 
ANTS_ratio = {}
for id_ in unique_ids:
    ANTS_ratio[id_] = ANTS_connectivity[id_] / (
        1. + ANTS_connectivity[id_ + '_total'])  

# make a DataFrame from it
ANTS_ratio = pd.DataFrame(ANTS_ratio)

# ANTS_ratio supposeldy contains some data that are ready for machine learning
