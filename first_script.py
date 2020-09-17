"""
This script reads the summaries of the imaging data to create
compact features for prediction.

Author: Bertrand Thirion, 2020
"""
#import libraries
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt
from math import pi

from sklearn.svm import SVC
from sklearn.model_selection import (cross_val_score, KFold, GridSearchCV,
                                     train_test_split)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import (DecisionTreeClassifier,export_text)


# replace with wherever you put that file
if 0:
    data_dir = '/Users/lebbe/Downloads/'
else:
    data_dir = os.getcwd()

# setting it only once to better adapt to local configuration 
csv_file = os.path.join(data_dir, 'all_connections.csv')
csv_file_behavior = os.path.join(data_dir, 'behavior2.csv')

# directory where figures are saved
write_dir = data_dir

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
# Keep only ANTS subjects

# ANTS = [subject for subject in unique_subjects if subject.endswith('ANTS')]
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
ANTS = [subject for subject in unique_subjects if subject.endswith('ANTS')]
ANTS_connectivity = average_connectivity[
    average_connectivity.subjectID.isin(ANTS)]
for id_ in unique_ids:
    ANTS_ratio[id_] = ANTS_connectivity[id_] / (
        1. + ANTS_connectivity[id_ + '_total'])

# make a DataFrame from it
ANTS_ratio = pd.DataFrame(ANTS_ratio)
ANTS_ratio.name = 'ANTS_ratio'
#transform data with sigmoid function 
ANTS_ratio_transformed = 1 / (1 + np.exp(np.asarray(- ANTS_ratio,dtype=float)))
ANTS_ratio_transformed = pd.DataFrame(ANTS_ratio_transformed)
ANTS_ratio_transformed.name = 'ANTS_ratio_transformed'
# ANTS_ratio supposeldy contains some data that are ready for machine learning
# do the same with FSL_connectivity
FSL_ratio = {}
for id_ in unique_ids:
    FSL_ratio[id_] = FSL_connectivity[id_] / (
        1. + FSL_connectivity[id_+'_total'])

# make a DataFrame from it : 
FSL_ratio = pd.DataFrame(FSL_ratio)
FSL_ratio.name = 'FSL_ratio'
#transform data with sigmoid function 
FSL_ratio_transformed = 1 / (1 + np.exp(np.asarray(- FSL_ratio,dtype=float)))
FSL_ratio_transformed = pd.DataFrame(FSL_ratio_transformed)
FSL_ratio_transformed.name = 'FSL_ratio_transformed'



##############################################################################
#LISTS ACCORDING TO BEHAVIOR
#change name of columns with abbreviations to make radar plots more readable
ANTS_ratio.columns = ANTS_ratio.columns.str.replace(r"CorticoThalamic","CT")
ANTS_ratio.columns = ANTS_ratio.columns.str.replace(r"CorticoStriatal","CS")
ANTS_ratio.columns = ANTS_ratio.columns.str.replace(r"CorticoCortical","CC")
#remove networks for which data are not reliable
ANTS_ratio_clean = ANTS_ratio.drop(["CC_1","CC_2","CC_3","CC_11","CS_1","CS_2",
                                    "CS_5","CS_6","CS_14","CT_1","CT_2","CT_3",
                                    "CT_4","CT_5","CT_6","CT_9","CT_10","CT_11",
                                    "CT_13","CT_15"], axis=1)
    
#change ANTS_ratio_clean index so that behavior and ANTS_ratio have the same index
ANTS_ratio_clean.index = behavior.index
ANTS_ratio_clean = ANTS_ratio_clean
frames = [ANTS_ratio_clean,behavior]
ANTS_ratio_clean = pd.concat(frames,axis=1)

#defines listes according to behavior of subjects ( cannot make a loop for this)
A = ANTS_ratio_clean.loc[ANTS_ratio_clean['eVP']== 0]
A.name = "A"
B = ANTS_ratio_clean.loc[ANTS_ratio_clean['eVP']== 1]
B.name = "B"
C = ANTS_ratio_clean.loc[ANTS_ratio_clean['lVP']== 0]
C.name = "C"
D = ANTS_ratio_clean.loc[ANTS_ratio_clean['lVP']== 1]
D.name = "D"
E = ANTS_ratio_clean.loc[ANTS_ratio_clean['siVP']== 0]
E.name = "E"
F = ANTS_ratio_clean.loc[ANTS_ratio_clean['siVP']== 1]
F.name = "F"
G = F.loc[F['eVP']== 0]
G.name = "G"
H = F.loc[F['eVP']== 1]
H.name = "H"
I = H.loc[F['lVP']== 0]
I.name = "I"
J = H.loc[F['lVP']== 1]
J.name = "J"

AB = A.append(B)
CD = C.append(D)
EF = E.append(F)
IJ = I.append(J)

liste_groupes = [A,B,C,D,E,F,G,H,I,J]


for i in liste_groupes : 
    i.means= i.iloc[:,0:26].mean()

means_AB =[A.means, B.means]
means_CD = [C.means,D.means]
means_EF = [E.means,F.means]
means_GH = [G.means, H.means]
means_IJ = [I.means,J.means]

#liste_comparaisons = [means_AB,means_CD,means_EF, means_GH, means_IJ]
#for i in liste_comparaisons
#    i = pd.concat (i,axis=1)
#    i = i.reset_index()
#    i["tvalue"]=""
#    i["pvalue"]=""
#
#    for j in i.index : 
#        i.loc[j,["tvalue"]]=mannwhitneyu(liste_groupes[i][means_AB.iloc[i,0]],
#                    B[means_AB.iloc[i,0]])[0]
#        means_AB.loc[i,["pvalue"]]=mannwhitneyu(A[means_AB.iloc[i,0]],
#                    B[means_AB.iloc[i,0]])[1]
#    means_AB["reject"]=multipletests(means_AB["pvalue"], alpha=0.05, 
#            method='fdr_bh', is_sorted=False, returnsorted=False)[0]
#    means_AB["p_corr"]=multipletests(means_AB["pvalue"], alpha=0.05, 
#            method='fdr_bh', is_sorted=False, returnsorted=False)[1]
#    
#    i.to_excel(r'/Users/lebbe/Downloads/means_AB_disco.xlsx')


##############################################################################
#PLOT
def make_spider2(row, title, liste, color, j, k):
    N = len(categories)
    # define the angle for each variable
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]   
    # initialize the spider plot
    ax = plt.subplot(7,5,row+1, polar=True, )
#
#    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1],
               [i for i in categories],
               color='black', size=6)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9],
               ["0.1", "0.3", "0.5", "0.7", "0.9"], color="grey", size=7)
    plt.ylim(0, 1)
     
    # Ind1
    values = liste.values[row].flatten().tolist()
    values += values[:1]
    ax.plot(angles[0:-1], values[j:k], color=color, linewidth=2,
            linestyle='solid')
    ax.fill(angles[0:-1], values[j:k], color=color, alpha=0.4)
   #add information regarding behavior
        #add information regarding siVP
    props_red = dict(boxstyle='circle', facecolor='red', alpha=0.5)
    props_green = dict(boxstyle='circle', facecolor='green', alpha=0.5)
    props_grey = dict(boxstyle='circle', facecolor='grey', alpha=0.5)
    bhv=[(26,"lVP",0.90),(28,"siVP",1),(29,"eVP",0.95)]
    for b,h,v in bhv :
        if liste.values[row,b]==1 : 
            
            ax.text(1, v, h + '+',transform=ax.transAxes,fontsize=2, 
                    verticalalignment = "top" ,bbox=props_red)
        elif liste.values[row,b]==0:
            
            ax.text(1, v, h +'-',transform=ax.transAxes,fontsize=2, 
                    verticalalignment = "top" ,bbox=props_green)
        else :
            
            ax.text(1, v, h + '?',transform=ax.transAxes,fontsize=2, 
                    verticalalignment = "top" ,bbox=props_grey)


    # Add a title
    plt.title(title, size=11, color='grey', y=1.1)
    
    plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.4)

def loop_to_plot2(liste, j, k):
    # initialize the figure
    my_dpi = 300
    plt.figure(figsize=(7000/my_dpi, 7000/my_dpi), dpi=my_dpi)
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(liste.index))
     
    # Give a name to big figure
    plt.gcf().text(0.9, 0.9, liste.name[1:], fontsize=40)
    # Loop to plot
    for row in range(0,len(liste.index)):
        make_spider2(row=row, liste = liste, title=liste.index[row].split(' ')[0], 
                    color=my_palette(row), j=j, k=k)
    # save figure 
    plt.savefig(write_dir + str(categories[0].partition('_')[0]) + '_' 
            + liste.name + '.png', dpi=None,
            facecolor='w', edgecolor='w', orientation='landscape',
            papertype=None, format=None,transparent=False, bbox_inches=None, 
            pad_inches=0.1, frameon=None, metadata=None)
# Figure

for letter in liste_groupes : 
    #define all columns with disconnection ratios in dataframe
    j = 0 
    k = 26
    liste=letter
    categories = list(liste)[j:k] 
    loop_to_plot2 (liste=liste, j=j, k=k)

##############################################################################
# PREDICT presence ofsevere intraoperative verbal perseverations (siVP) 
# by applying different supervised learning analysis



#select disconnectome data : 

X = EF
y = EF["siVP"]


####SVC rbf
nested_score_rbf=np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
svm = SVC(kernel="rbf") 
# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
nested_score_rbf = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of {:6f} with std. dev. of {:6f}.'
      .format(nested_score_rbf.mean(),nested_score_rbf.std()))

###SVC linear
nested_score_linear=np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100],
          "gamma": [.01, .1]}
svm = SVC(kernel="linear") 
# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
nested_score_linear = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'
      .format(nested_score_linear.mean(),nested_score_linear.std()))

###random forest
nested_score_rf = np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"max_depth": [2, 4, 6, 8], "n_estimators" : 
    [2,4,6,8,10,12,14,16,18,20]}
rf= RandomForestClassifier(random_state=0)

# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv)
nested_score_rf = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'
      .format(nested_score_rf.mean(),nested_score_rf.std()))

###Logistic Regression
nested_score_lr = np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"C": [1, 10, 100]}
lr= LogisticRegression(random_state=0)

# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=lr, param_grid=p_grid, cv=inner_cv)
nested_score_lr = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'
      .format(nested_score_lr.mean(),nested_score_lr.std()))

#####TREE

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
tree.plot_tree(clf)
clf.score(X_test, y_test)

r = export_text(clf,X.columns)
print(r)

#NESTED CROSSVALIDATION
nested_score_rf = np.zeros(4)
# Set up possible values of parameters to optimize over
p_grid = {"max_depth": [2, 4, 6, 8], "min_samples_leaf" : [1,2,3,4,5]}
rf= tree.DecisionTreeClassifier(random_state=0)

# crossvalidation
inner_cv = KFold(n_splits=4, shuffle=True, random_state=0)
outer_cv = KFold(n_splits=4, shuffle=True, random_state=0)
# Nested CV with parameter optimization
clf = GridSearchCV(estimator=rf, param_grid=p_grid, cv=inner_cv)
nested_score_tree = cross_val_score(clf, X=X, y=y, cv=outer_cv)
print('Mean score of of {:6f} with std. dev. of {:6f}.'.format(nested_score_rf.mean(),nested_score_rf.std()))
r = export_text(clf,X.columns)
print(r)   
