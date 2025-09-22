# -*- coding: utf-8 -*-
"""
Spyder Editor
@Yelyzaveta Snihirova
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import median
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, roc_curve, auc, roc_auc_score, recall_score,confusion_matrix,balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler
import nibabel as nib
import os 
from skimage.measure import regionprops, label, regionprops_table
from untitled0 import create_output
import glob
from nilearn._utils import check_niimg
from collections import Counter, defaultdict
import scipy.spatial.distance
from scipy.stats import iqr
#
#Change types of columns
#
#
raw_data = pd.read_csv(r'Z:\ritter\share\data\MS\deepMS\CIS_merged_20201030.csv')
image_dir = 'Z:/ritter/share/data/MS/deepMS/CIS'
data_dir = 'Z:/ritter/share/data/MS/deepMS/'
raw_data_lesions = pd.read_csv(r'Z:\ritter\share\data\MS\deepMS\Lesions_SIENAX_20201030.csv')
df = raw_data
df = df.replace('missing', np.nan)
df = df.sort_values(['subject', 'mri_date', 'bids_session'])
raw_data_lesions = raw_data_lesions.sort_values(['Subject', 'mri_date'])

for subject in df['subject']:
    idx = df[df['subject'] == subject].index
    visit_list = os.listdir(os.path.join(image_dir,subject))
    visit_list.sort()
    for i, sub_idx in enumerate(idx): 
        name_flair_stripped = 'BET_' + subject + '_' + visit_list[i] + '_FLAIR.nii.gz'    
        name_flair = subject + '_' + visit_list[i] + '_FLAIR.nii.gz'
        name_mprage = subject + '_' + visit_list[i] + '_MPRAGE.nii.gz'
        name_lmask = subject + '_' + visit_list[i] + '_T2LESION_QC.nii.gz'
        path_flair_stripped = data_dir + 'CIS/' + subject + '/' + visit_list[i] + '/anat/' + name_flair_stripped
        path_flair = data_dir + 'CIS/' + subject + '/' + visit_list[i] + '/anat/' + name_flair
        path_mprage = data_dir + 'CIS/' + subject + '/' + visit_list[i] + '/anat/' + name_mprage
        path_lmask = data_dir + 'CIS/' + subject + '/' + visit_list[i] + '/anat/' + name_lmask
        df.loc[sub_idx, 'path_flair_stripped'] = path_flair_stripped
        df.loc[sub_idx, 'path_flair'] = path_flair
        df.loc[sub_idx, 'path_mprage'] = path_mprage
        df.loc[sub_idx, 'path_mask'] = path_lmask 
# format: e.g. path + subject_id + sess + anat + subject_id_sess_FLAIR.nii.gz

df['t25fw1'] = df['t25fw1'].astype(float)
df['t25fw2'] = df['t25fw2'].astype(float)
df['weight'] = df['weight'].astype(float)
df['weight_kg'] = df['weight_kg'].str.replace(',','.')
df['weight_kg'] = df['weight_kg'].astype(float)
df['height_cm'] = df['height_cm'].astype(float)
df['bmi'] = df['bmi'].astype(float)
df['cigarettes'] = df['cigarettes'].astype(float)
df['dom_hand1'] = df['dom_hand1'].astype(float)
df['dom_hand2'] = df['dom_hand2'].astype(float)
df['nondom_hand1'] = df['nondom_hand1'].astype(float)
df['nondom_hand2'] = df['nondom_hand2'].astype(float)

df['oral_steroids'] = df['oral_steroids'].astype('category')
df['oral_steroids.1'] = df['oral_steroids.1'].astype('category')
df['casual_smoker'] = df['casual_smoker'].astype('category')
df['fu_therapy'] = df['fu_therapy'].astype('category')
df['fu_therapy.1'] = df['fu_therapy.1'].astype('category')
df['smoke_last6mths'] = df['smoke_last6mths'].astype('category')
df['vitD_use'] = df['vitD_use'].astype('category')
df['dom_hand'] = df['dom_hand'].astype('category')
df['sex'] = df['sex'].astype('category')
df['subject'] = df['subject'].astype('category')
df['ms_therapy_stopped'] = df['ms_therapy_stopped'].astype('category')
df['visual_fs'] = df['visual_fs'].astype('category')
df['pyramidal_fs'] = df['pyramidal_fs'].astype('category')
df['brainstem_fs'] = df['brainstem_fs'].astype('category')
df['cerebellum_fs'] = df['cerebellum_fs'].astype('category')
df['sensory_fs'] = df['sensory_fs'].astype('category')
df['cerebral_fs'] = df['cerebral_fs'].astype('category')
df['bowelbladder_fs'] = df['bowelbladder_fs'].astype('category')
df['ambulation_fs'] = df['ambulation_fs'].astype('category')
df['diagnosis_mcd2017'] = df['diagnosis_mcd2017'].astype('category')

df['mri_date'] = pd.to_datetime(df['mri_date'])
df['clinical_visit'] = pd.to_datetime(df['clinical_visit'])
df['end_date_treatment'] = pd.to_datetime(df['end_date_treatment'])
df['start_date_escalation'] = pd.to_datetime(df['start_date_escalation'])
df['start_date_treatment'] = pd.to_datetime(df['start_date_treatment'])
df['start_date_symptoms'] = pd.to_datetime(df['start_date_symptoms'])
df['end_date_treatment'] = pd.to_datetime(df['end_date_treatment'])
df['start_date_escalation'] = pd.to_datetime(df['start_date_escalation'])
df['full_remission'] = pd.to_datetime(df['full_remission'])
df['start_date_symtom2'] = pd.to_datetime(df['start_date_symtom2'])
df['start_date_treatment.1'] = pd.to_datetime(df['start_date_treatment.1'])
df['end_date_treatment.1'] = pd.to_datetime(df['end_date_treatment.1'])
df['start_date_escalation.1'] = pd.to_datetime(df['start_date_escalation.1'])
df['end_date_escalation.1'] = pd.to_datetime(df['end_date_escalation.1'])
df['lastday_oralsteroids.1'] = pd.to_datetime(df['lastday_oralsteroids.1'])
df['start_date_symtom3'] = pd.to_datetime(df['start_date_symtom3'])
df['start_date_treatment.2'] = pd.to_datetime(df['start_date_treatment.2'])
df['end_date_treatment.2'] = pd.to_datetime(df['end_date_treatment.2'])
df['start_date_escalation.2'] = pd.to_datetime(df['start_date_escalation.2'])
df['end_date_escalation.2 '] = pd.to_datetime(df['end_date_escalation.2'])
df['lastday_oralsteroids.2'] = pd.to_datetime(df['lastday_oralsteroids.2'])
df['ms_therapy_enddate'] = pd.to_datetime(df['ms_therapy_enddate'])
df['other_ms_meds_startdate'] = pd.to_datetime(df['other_ms_meds_startdate'])
df['other_ms_meds_enddate'] = pd.to_datetime(df['other_ms_meds_enddate'])
df['edss_date'] = pd.to_datetime(df['edss_date'])
df['ms_therapy_startdate'] = df['ms_therapy_startdate'].astype(object)
df['ms_therapy_startdate'] = df['ms_therapy_startdate'].replace('0411-05-02',None)
df['ms_therapy_startdate'] = pd.to_datetime(df['ms_therapy_startdate'])
df['lastday_oralsteroids'] = pd.to_datetime(df['lastday_oralsteroids'])
# 
#
#Save patients with MRI data +2 years +/- 6 months from baseline
#
#
patient_min_time = df.groupby('subject', as_index=False).agg({'mri_date':'min'}).rename({'mri_date':'min_mri_date'} , axis=1)
df= df.merge(patient_min_time, how='outer')
df['t25wb_average'] = df[['t25fw1','t25fw2']].mean(axis = 1)
df['dom_hand_average'] = df[['dom_hand1','dom_hand2']].mean(axis = 1)
df['nondom_hand_average'] = df[['nondom_hand1','nondom_hand2']].mean(axis = 1)
df['2years+6month'] = df['min_mri_date'] + pd.DateOffset(months=30)
df['2years-6month'] = df['min_mri_date'] + pd.DateOffset(months = 18)
df_final = pd.DataFrame()
df_final = df[((df['mri_date'] <= df['2years+6month']) | (df['mri_date'] == df['min_mri_date']))&(df['mri_date'] >= df['2years-6month']) | (df['mri_date'] == df['min_mri_date'])]
#delete patients with just one visit
df_final = df_final.groupby('subject').filter(lambda x: len(x) > 1)
#delete first patient with baseline not equal to 1st visit
df_final = df_final.iloc[2:]
#delete two repeated MRI in this range 
df_final = df_final[df_final.mri_date != '2013-12-11']
df_final = df_final[df_final.mri_date != '2014-08-26']
# #
# #
# #Get data with just baseline data
# #
# #
df_baseline = pd.DataFrame()
df_baseline =  df_final.loc[df['event_name'] == 'Baseline']
#
#
## Graphs
#
#
#Histogram for age and sex
#
#
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

age_sex_graph = sns.FacetGrid(df_baseline,hue='sex')
age_sex_graph = age_sex_graph.map(sns.kdeplot,'age_onset',shade=True,alpha=0.4)                      
plt.legend(labels=['Male', 'Female'])
plt.title('Distribution of Age and Sex')
plt.xlabel("Age of Onset")
plt.show()
plt.close()
#
#
##EDSS and Timed 25-Foot Walk-test scores for sex
#
#
scatter_diagn_scores = sns.relplot(x='edss_score', y='t25wb_average', hue='diagnosis_mcd2017',
              palette="bright",
            height=6, data=df_baseline, s = 50)
plt.legend(labels=['RRMS', 'CIS'])
scatter_diagn_scores._legend.remove()
plt.title('EDSS and Timed 25-Foot Walk-test scores for RRMS and CIS')
plt.ylabel("T25FW")
plt.xlabel("EDSS")
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.show()
plt.close()
#
#
#EDSS and Timed 25-Foot Walk-test scores for Male and Female scatter
#
#
scatter_sex_scores = sns.relplot(x='edss_score', y='t25wb_average', hue='sex',
              palette="Paired",
            height=6, data=df_baseline, s = 50)
plt.legend(labels=['Female', 'Male'])
scatter_sex_scores._legend.remove()
plt.title('EDSS and Timed 25-Foot Walk-test scores for Male and Female')
plt.ylabel("T25FW")
plt.xlabel("EDSS")
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.show()
plt.close()
#
#
# Some statistics
#
#
def stats (column):
    print('Mean: ',column.mean())
    print('Std: ',column.std())
    print('Max: ',column.max())
    print('Min: ',column.min())
    print('Median: ', median(column))    
    print('25th percentile:', np.nanpercentile(column, 25))
    print('75th percentile:', np.nanpercentile(column, 75))
    print('IQR:', np.subtract(*np.nanpercentile(column, [75, 25])))
# discribtion_of_data = df_final.describe()
# missing_values = pd.isnull(df_final).sum()
# df_final['sex'].value_counts()
#
#
##Lesions data obtaining, getting just baseline and difference between baseline and +2 years
#
#
data_lesions = raw_data_lesions.loc[df_final.index]

data_lesions_cut = data_lesions[['Subject', 'manual_t2c_all', 'manual_t2v_all']]
data_lesions_cut['edss_score'] = df_final['edss_score'].values
data_lesions_cut['t25wb_average'] = df_final['t25wb_average'].values
data_lesions_cut['dom_hand_average'] = df_final['dom_hand_average'].values
data_lesions_cut['nondom_hand_average'] = df_final['nondom_hand_average'].values
data_lesions_cut_difference =  data_lesions_cut.groupby('Subject').apply(lambda x: x.iloc[1]-x.iloc[0])
data_lesions_cut['event_name'] = df_final['event_name'].values
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6885577/
data_lesions_cut['dom_hand_average_20'] = data_lesions_cut['dom_hand_average'].values/5
data_lesions_cut['nondom_hand_average_20'] = data_lesions_cut['nondom_hand_average'].values/5
data_lesions_cut_baseline = data_lesions_cut.loc[data_lesions_cut['event_name'] == 'Baseline']
#
#
#Condition probability calculation
# edss = bigger is worse, positive value of difference between 2 years and baseline is bad
# t25 = bigger is worse, positive is bad
# hand = bigger is worse, positive is bad 
#
#
#get equal indices = subjects
data_lesions_cut_baseline_index = data_lesions_cut_baseline.set_index('Subject')
data_lesions_cut_difference_copy = pd.DataFrame()
data_lesions_cut_difference_copy = data_lesions_cut_difference
data_lesions_cut_difference_copy['edss_score_initial'] = data_lesions_cut_baseline_index['edss_score']
#DOI: 10.1177/1352458517709619
data_lesions_cut_difference_copy['edss_score'] = np.where((data_lesions_cut_difference_copy.edss_score_initial >0)&(data_lesions_cut_difference_copy.edss_score_initial <= 5) & (data_lesions_cut_difference_copy.edss_score>=1)|(data_lesions_cut_difference_copy.edss_score_initial >= 5.5)& (data_lesions_cut_difference_copy.edss_score>=0.5)|(data_lesions_cut_difference_copy.edss_score_initial ==0)& (data_lesions_cut_difference_copy.edss_score>=1.5),'1', '0')

# #save data for patiens with no nan data 
data_lesions_cut_difference_copy['t25wb_average_initial'] = data_lesions_cut_baseline_index['t25wb_average']
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6885577/
data_lesions_cut_difference_copy['t25wb_average_20'] = data_lesions_cut_difference_copy['t25wb_average'].values/5
data_lesions_cut_index = data_lesions_cut.set_index('Subject')
data_lesions_cut_index_baseline = data_lesions_cut_index.loc[data_lesions_cut_index['event_name'] == 'Baseline']
data_lesions_cut_difference_copy['t25wb_average'] = np.where(data_lesions_cut_difference_copy['t25wb_average']>data_lesions_cut_difference_copy['t25wb_average_20'],'1', '0')
data_lesions_cut_difference_copy['dom_hand_average_initial'] = data_lesions_cut_baseline_index['dom_hand_average']
data_lesions_cut_difference_copy['nondom_hand_average_initial'] = data_lesions_cut_baseline_index['nondom_hand_average']
data_lesions_cut_difference_copy['dom_hand_average_20'] = data_lesions_cut_index_baseline['dom_hand_average_20']
data_lesions_cut_difference_copy['nondom_hand_average_20'] = data_lesions_cut_index_baseline['nondom_hand_average_20']
data_lesions_cut_difference_copy['dom_hand_average'] = np.where(data_lesions_cut_difference_copy['dom_hand_average']>data_lesions_cut_difference_copy['dom_hand_average_20'],'1', '0')
data_lesions_cut_difference_copy['nondom_hand_average'] = np.where(data_lesions_cut_difference_copy['nondom_hand_average']>data_lesions_cut_difference_copy['nondom_hand_average_20'],'1', '0')
data_lesions_cut_difference_copy['manual_t2c_all_diff'] = data_lesions_cut_difference_copy['manual_t2c_all']
data_lesions_cut_difference_copy['manual_t2c_all'] = np.where(data_lesions_cut_difference_copy['manual_t2c_all']<0,'1', '0')

#
#
#

data_lesions_cut_difference_copy.dropna(inplace=True)
data_lesions_cut_baseline_index = data_lesions_cut_baseline_index.loc[data_lesions_cut_difference_copy.index]

def mnb_model (X, y, name, i): 
    y=y.astype('int')   
    conf_matrix_list_of_arrays = []
    acc, prec, rec, bacc, auc_sc  = list(), list(), list(), list(), list()       
    cv =KFold(n_splits=5, random_state=i, shuffle = True)
    for train_ix, test_ix in cv.split(X, y):
        print("TRAIN:", train_ix, "TEST:", test_ix)
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]
        oversample = RandomOverSampler()
        X_train_fold_upsample, y_train_fold_upsample = oversample.fit_resample(X_train, y_train)
        mnb = MultinomialNB().fit(X_train_fold_upsample, y_train_fold_upsample)
        y_pred = mnb.predict(X_test)
        y_pred_prob = mnb.predict_proba(X_test)[:,1]  
        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred, average="binary", pos_label=0))
        rec.append(recall_score(y_test, y_pred, average="binary", pos_label=0))
        bacc.append(balanced_accuracy_score(y_test, y_pred))

        try:
            auc_sc.append(roc_auc_score(y_test, y_pred_prob))
        except ValueError:
            auc_sc.append(None)
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_list_of_arrays.append(conf_matrix)
      #number of correct predictions made divided by the total number of predictions made, multiplied by 100 to turn it into a percentage
    print('Accuracy score mean, std: {0:0.4f}'. format(np.mean(acc), np.std(acc)))
    #tp / (tp + fp). The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    print('Precision score mean, std: ', np.mean(prec), np.std(prec))
    #tp / (tp + fn) The recall is intuitively the ability of the classifier to find all the positive samples.
    print('Recall score (Sensitivity) mean, std: ', np.mean(rec), np.std(rec))
    #The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class
    print('Balanced Accuracy score mean, std: ', np.mean(bacc), np.std(bacc))
    mean_of_conf_matrix_arrays = np.mean(conf_matrix_list_of_arrays, axis=0)    
    tn, fp, fn, tp = mean_of_conf_matrix_arrays.ravel()
        #Specificity measures the proportion of negatives that are correctly identified (i.e. the proportion of those who do not have the condition (unaffected) who are correctly identified as not having the condition).
    print('Specificity: ', tn / (tn+fp))  
    cm = confusion_matrix(y_test, y_pred)      
    classification_error = (fp + fn) / float(tp + tn + fp + fn)
    
    print('Classification error : {0:0.4f}'.format(classification_error))
    print('Confusion matrix\n\n', cm)
    print('\nTrue Positives(TP) = ', cm[0,0])    
        
    if len(cm)>1:
        print('\nTrue Negatives(TN) = ', cm[1,1])
        print('\nFalse Positives(FP) = ', cm[0,1])
        print('\nFalse Negatives(FN) = ', cm[1,0])
        tr_neg = cm[1,1]
        fals_pos = cm[0,1]
        fals_neg = cm[1,0]
        cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                  index=['Predict Positive:1', 'Predict Negative:0'])

        sns.heatmap(cm_matrix, square=True, annot=True, cmap='RdBu', cbar=False, fmt = 'd')
        plt.show()
        
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,4))
    
        plt.plot(fpr, tpr, linewidth=2)
        
        plt.plot([0,1], [0,1], 'k--' )
        
        plt.rcParams['font.size'] = 12
        plt.plot(fpr, tpr, label='AUC = ' + str(round(roc_auc, 2)))
        plt.legend(loc='lower right')
        
        plt.title('ROC curve for Multinomial Naive Bayes Classifier')
        
        plt.xlabel('False Positive Rate (1 - Specificity)')
            
        plt.ylabel('True Positive Rate (Sensitivity)')
    
        plt.show()
    else:
        tr_neg = None
        fals_neg = None
        fals_pos = None
            
            
    true_positive_rate = tp / (tp + fn)
    
    
    print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
    false_positive_rate = fp / float(fp + tn)
    
    
    print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
       
    column = {name:[np.mean(acc), np.mean(prec), np.mean(rec), tn / (tn+fp), np.mean(bacc), classification_error, cm[0,0], tr_neg, fals_pos, fals_neg, true_positive_rate, false_positive_rate]}  
    stats_table = pd.DataFrame(column, index =['Accuracy mean', 'Precision score', 'Recall score (Sensitivity)', 'Specificity', 'Balanced Accuracy score',  'Classification error', 'True Positives(TP)','True Negatives(TN)','False Positives(FP)','False Negatives(FN)', 'True Positive Rate','False Positive Rate'])    
    return stats_table


def gnb_model(X1,y1,name1, i1):
    y1=y1.astype('int')   
    conf_matrix_list_of_arrays1 = []
    acc1, prec1, rec1, bacc1,  auc_sc1  = list(), list(), list(), list(), list()       
    cv1 = KFold(n_splits=5, random_state=i1, shuffle = True)
    for train_ix1, test_ix1 in cv1.split(X1, y1):
        X_train1, X_test1 = X1.iloc[train_ix1], X1.iloc[test_ix1]
        y_train1, y_test1 = y1.iloc[train_ix1], y1.iloc[test_ix1]
        oversample1 = RandomOverSampler()
        X_train_fold_upsample1, y_train_fold_upsample1 = oversample1.fit_resample(X_train1, y_train1)
        gnb = GaussianNB().fit(X_train_fold_upsample1, y_train_fold_upsample1)
        y_pred1 = gnb.predict(X_test1)
        y_pred_prob1 = gnb.predict_proba(X_test1)[:,1]  
        acc1.append(accuracy_score(y_test1, y_pred1))
        prec1.append(precision_score(y_test1, y_pred1, average="binary", pos_label=0))
        rec1.append(recall_score(y_test1, y_pred1, average="binary", pos_label=0))
        bacc1.append(balanced_accuracy_score(y_test1, y_pred1))
        try:
            auc_sc1.append(roc_auc_score(y_test1, y_pred_prob1))
        except ValueError:
            auc_sc1.append(None)
        conf_matrix1 = confusion_matrix(y_test1, y_pred1)
        conf_matrix_list_of_arrays1.append(conf_matrix1)
      #number of correct predictions made divided by the total number of predictions made, multiplied by 100 to turn it into a percentage
    print('Model accuracy score mean, std: {0:0.4f}'. format(np.mean(acc1), np.std(acc1)))
    #tp / (tp + fp). The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    print('Precision score mean, std: ', np.mean(prec1), np.std(prec1))
    #tp / (tp + fn) The recall is intuitively the ability of the classifier to find all the positive samples.
    print('Recall score (Sensitivity) mean, std: ', np.mean(rec1), np.std(rec1))
    #The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class
    print('Balanced Accuracy score mean, std: ', np.mean(bacc1), np.std(bacc1))
    mean_of_conf_matrix_arrays1 = np.mean(conf_matrix_list_of_arrays1, axis=0)    
    tn1, fp1, fn1, tp1 = mean_of_conf_matrix_arrays1.ravel()
        #Specificity measures the proportion of negatives that are correctly identified (i.e. the proportion of those who do not have the condition (unaffected) who are correctly identified as not having the condition).
    print('Specificity: ', tn1 / (tn1+fp1))  
    cm1 = confusion_matrix(y_test1, y_pred1)      
    classification_error1 = (fp1 + fn1) / float(tp1 + tn1 + fp1 + fn1)
    
    print('Classification error : {0:0.4f}'.format(classification_error1))
    print('Confusion matrix\n\n', cm1)
    print('\nTrue Positives(TP) = ', cm1[0,0])    
        
    if len(cm1)>1:
        print('\nTrue Negatives(TN) = ', cm1[1,1])
        print('\nFalse Positives(FP) = ', cm1[0,1])
        print('\nFalse Negatives(FN) = ', cm1[1,0])
        tr_neg1 = cm1[1,1]
        fals_pos1 = cm1[0,1]
        fals_neg1 = cm1[1,0]
        cm_matrix1 = pd.DataFrame(data=cm1, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                  index=['Predict Positive:1', 'Predict Negative:0'])

        sns.heatmap(cm_matrix1, square=True, annot=True, cmap='RdBu', cbar=False, fmt = 'd')
        plt.show()
        
        fpr1, tpr1, _ = roc_curve(y_test1, y_pred1)
        roc_auc1 = auc(fpr1, tpr1)
        plt.figure(figsize=(6,4))
    
        plt.plot(fpr1, tpr1, linewidth=2)
        
        plt.plot([0,1], [0,1], 'k--' )
        
        plt.rcParams['font.size'] = 12
        plt.plot(fpr1, tpr1, label='AUC = ' + str(round(roc_auc1, 2)))
        plt.legend(loc='lower right')
        
        plt.title('ROC curve for Gaussian Naive Bayes Classifier')
        
        plt.xlabel('False Positive Rate (1 - Specificity)')
            
        plt.ylabel('True Positive Rate (Sensitivity)')
    
        plt.show()
    else:
        tr_neg1 = None
        fals_neg1 = None
        fals_pos1 = None
            
            
    true_positive_rate1 = tp1 / (tp1 + fn1)
    
    
    print('True Positive Rate : {0:0.4f}'.format(true_positive_rate1))
    false_positive_rate1 = fp1 / float(fp1 + tn1)
    
    
    print('False Positive Rate : {0:0.4f}'.format(false_positive_rate1))
       
    column = {name1:[np.mean(acc1), np.mean(prec1), np.mean(rec1), tn1 / (tn1+fp1), np.mean(bacc1), classification_error1, cm1[0,0], tr_neg1, fals_pos1, fals_neg1, true_positive_rate1, false_positive_rate1]}  
    stats_table = pd.DataFrame(column, index =['Model accuracy score', 'Precision score', 'Recall score (Sensitivity)', 'Specificity', 'Balanced Accuracy score', 'Classification error', 'True Positives(TP)','True Negatives(TN)','False Positives(FP)','False Negatives(FN)', 'True Positive Rate','False Positive Rate'])    
    return stats_table

#
#
def model_made(column1, column2, name, name_model):
    model =[]
    for i in range(0,5):
        model.append(name_model(column1, column2, name, i))
    model = pd.concat(model, axis = 1)
    model[name] = model.mean(axis=1)
    model = model.iloc[:, 0]
    return model
mnb_table = pd.concat([model_made(data_lesions_cut_baseline_index[['manual_t2c_all']], data_lesions_cut_difference_copy['edss_score'], 'Number of lesions vs EDSS score', mnb_model),
model_made(data_lesions_cut_baseline_index[['manual_t2c_all']], data_lesions_cut_difference_copy['manual_t2c_all'], 'Number of lesions vs Number of lesions in 2 years', mnb_model), 
model_made(data_lesions_cut_baseline_index[['manual_t2c_all']], data_lesions_cut_difference_copy['nondom_hand_average'], 'Number of lesions vs 9HPT score (non-dominant hand)', mnb_model), 
model_made(data_lesions_cut_baseline_index[['manual_t2c_all']], data_lesions_cut_difference_copy['dom_hand_average'], 'Number of lesions vs 9HPT score (dominant hand)', mnb_model),
model_made(data_lesions_cut_baseline_index[['manual_t2c_all']], data_lesions_cut_difference_copy['t25wb_average'], 'Number of lesions vs T25FW score', mnb_model)], join='outer', axis=1)

gnb_table = pd.concat([model_made(data_lesions_cut_baseline_index[['manual_t2v_all']], data_lesions_cut_difference_copy['edss_score'], 'Volume of lesions vs EDSS score', gnb_model),
model_made(data_lesions_cut_baseline_index[['manual_t2v_all']], data_lesions_cut_difference_copy['manual_t2c_all'], 'Volume of lesions vs Number of lesions in 2 years', gnb_model),
model_made(data_lesions_cut_baseline_index[['manual_t2v_all']], data_lesions_cut_difference_copy['nondom_hand_average'], 'Volume of lesions vs 9HPT score (non-dominant hand)', gnb_model),
model_made(data_lesions_cut_baseline_index[['manual_t2v_all']], data_lesions_cut_difference_copy['dom_hand_average'], 'Volume of lesions vs 9HPT score (dominant hand)', gnb_model),
model_made(data_lesions_cut_baseline_index[['manual_t2v_all']], data_lesions_cut_difference_copy['t25wb_average'], 'Volume of lesions vs T25FW score', gnb_model)], join='outer', axis=1)



#load nifti to 3D nparray
def load_nifti(file_path, z_factor=None, dtype=None, incl_header=False, mask=None):
    if dtype is None:
        dt = np.float32  
    else:
        dt = dtype
    img = nib.load(file_path)
    struct_arr = img.get_fdata().astype(dt)
    if mask is not None:
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = nib.interpolation.zoom(struct_arr, z_factor)
    if incl_header:
        return struct_arr, img
    else:
        return struct_arr
#
#
# Manual checking of lesions count in all brain 
#
#

#list_of_lesions_number = []
# for i in list(df_baseline['path_mask']):
#         image = load_nifti(i)        
#         img=label(image, background=0)
#         img=img+1
#         propsa = regionprops(img)
#         length = len(propsa)
#         list_of_lesions_number.append(str(length-1))
epi_img = load_nifti(r'Z:/ritter/share/data/MS/deepMS/CIS/CIS_0003B4/sess_01/anat/CIS_0003B4_sess_01_T2LESION_QC.nii.gz')
anatomy_img = load_nifti(r'Z:\ritter\share\misc\masks\FSL_atlases\JHU\JHU-ICBM-labels-1mm.nii.gz')#
#
#
# Manual checking of lesions count in specific region brain    
#
#     
region = np.where(anatomy_img==4, 1, 0)
image = region*epi_img
img=label(image, background=0)
img=img+1
propsa = regionprops(img)
length = len(propsa)
print(str(length-1))
#
#
# Dividing lesions into big and small 
#
#
lesions_count_by_type = defaultdict(list)
for i in list(df_baseline['path_mask']):
        image = load_nifti(i)        
        img=label(image, background=0)
        img=img+1
        table = regionprops_table(img,properties=['area'])
        l=[]
        [l.extend([k,v]) for k,v in table.items()]
        l = l[1]
        l = list(l[1:])

        medium = sum(float(num) >= 26 for num in l)

        small = sum(float(num) >= 3 and float(num)<=25 for num in l)


        values = [small, medium]
        for item in values:
            lesions_count_by_type[i].append(item)
lesions_count_by_type_df = pd.DataFrame.from_dict(lesions_count_by_type, orient='index')
lesions_count_by_type_df.index = [x[63:77] for x in lesions_count_by_type_df.index]
lesions_count_by_type_df.columns = ['small', 'medium']
lesions_count_by_type_df.index = sorted(lesions_count_by_type_df.index.values, key=lambda s: s.lower())
data_lesions_cut_difference_copy.index = [x[4:] for x in data_lesions_cut_difference_copy.index]
lesions_count_by_type_df_copy = lesions_count_by_type_df
lesions_count_by_type_df_copy.index = [x[:6] for x in lesions_count_by_type_df_copy.index]
lesions_count_by_type_df_copy_model = lesions_count_by_type_df_copy.loc[data_lesions_cut_difference_copy.index]

mnb_table_type_lesions = pd.concat([model_made(lesions_count_by_type_df_copy_model[['small']], data_lesions_cut_difference_copy['edss_score'], 'Number of small lesions vs EDSS score', mnb_model),
                                    model_made(lesions_count_by_type_df_copy_model[['medium']], data_lesions_cut_difference_copy['edss_score'], 'Number of medium lesions vs EDSS score', mnb_model),
                                    model_made(lesions_count_by_type_df_copy_model[['small']], data_lesions_cut_difference_copy['t25wb_average'], 'Number of small lesions vs T25FW score', mnb_model),
                                    model_made(lesions_count_by_type_df_copy_model[['medium']], data_lesions_cut_difference_copy['t25wb_average'], 'Number of medium lesions vs T25FW score', mnb_model),
                                    model_made(lesions_count_by_type_df_copy_model[['small']], data_lesions_cut_difference_copy['nondom_hand_average'], 'Number of small lesions vs 9HPT score (non-dominant hand)', mnb_model),
                                    model_made(lesions_count_by_type_df_copy_model[['medium']], data_lesions_cut_difference_copy['nondom_hand_average'], 'Number of medium lesions vs 9HPT score (non-dominant hand)', mnb_model),
                                    model_made(lesions_count_by_type_df_copy_model[['small']], data_lesions_cut_difference_copy['dom_hand_average'], 'Number of small lesions vs 9HPT score (dominant hand)', mnb_model),
                                    model_made(lesions_count_by_type_df_copy_model[['medium']], data_lesions_cut_difference_copy['dom_hand_average'], 'Number of medium lesions vs 9HPT score (dominant hand)', mnb_model)], join='outer', axis=1)
#
#
#create tables with data of lesions
#
#
# for i in df_baseline.iloc[0:13]['path_mask']:
#     create_output(i, direction = 'both', voxel_thresh=0.1, cluster_extent=0.5,outdir='C:\\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold')
#
def get_largest_or_second_largest_percentage(string):
    """Given a string of data, return the largest percentage, 
    the second-largest if the largest has label 'no_label',
    the only percentage if there is one data point
    """
    parts = [p.strip() for p in string.split(";")]
    # If there is only one item, don't bother sorting.
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    # Sort from lowest to highest percentage.
    # Assumes that the percentage is before the % symbol.
    parts.sort(key=lambda s: float(s.split("%")[0]))
    # Return second-largest if the largest has label 'no_label'.
    if "no_label" in parts[-1]:
        return parts[-2]
    return parts[-1]  # return largest otherwise.

def make_table_for_specific_lesions(files_path, atlasname, labels_path):
    """Read all the files from folder after create_output was performed. 
    It makes 1) a table with percentage with nonidentified lesions for each
    patient: 2) a table with number of lesions for patients per anatomical area
    3) table of get_largest_or_second_largest_percentage applied for all columns
    """
    all_files = glob.glob(os.path.join(files_path, "*_clusters.csv"))
    all_files = sorted(all_files)
    names = [x[61:75] for x in all_files]
    dataframes = [pd.read_csv(p) for p in all_files]
    merged_dataframe = pd.concat(dataframes, axis=1)
    column = merged_dataframe[[atlasname]]
    column = column.fillna('NA').applymap(get_largest_or_second_largest_percentage)     
    column = column.replace('NA', np.nan)
    Perc_of_nonind = 100*(((column[atlasname] == '100.00% no_label')*1).sum()/(len(column.index)-(column.isnull()*1).sum()))
    column.columns = names
    Perc_of_nonind.index = names
    atlas_file = pd.read_csv(labels_path)
    atlas_file = atlas_file.drop_duplicates('name')
    keyword_dict = {w:w.replace(' ', '_') for w in atlas_file['name']}
    final_counts = []
    for u in list(column.columns):
        corpus = ' '.join(column[u].fillna(''))
        for w,w2 in keyword_dict.items():
            corpus = corpus.replace(w,w2)    
        all_counts = Counter(corpus.split())
        final_counts.append({w:all_counts[w2] for w,w2 in keyword_dict.items()})
    final_counts_df = pd.DataFrame.from_records(final_counts)
    final_counts_df.index = list(names)
    return(Perc_of_nonind, final_counts_df, column)


final_counts_jhutracs_df = make_table_for_specific_lesions(r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold' , 'jhutracs', r'C:\Users\Public\Documents\Conda\Lib\site-packages\atlasreader\data\atlases\labels_jhutracs.csv')[1]
final_counts_jhu_df = make_table_for_specific_lesions(r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold' , 'jhuatlas1mm', r'C:\Users\Public\Documents\Conda\Lib\site-packages\atlasreader\data\atlases\labels_jhuatlas1mm.csv')[1]
final_counts_oxfcort1mm_df = make_table_for_specific_lesions(r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold' , 'oxfcort1mm', r'C:\Users\Public\Documents\Conda\Lib\site-packages\atlasreader\data\atlases\labels_oxfcort1mm.csv')[1]
final_counts_oxfsubcort1mm_df = make_table_for_specific_lesions(r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold' , 'oxfsubcort1mm', r'C:\Users\Public\Documents\Conda\Lib\site-packages\atlasreader\data\atlases\labels_oxfsubcort1mm.csv')[1]
final_counts_neuromorph_df = make_table_for_specific_lesions(r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold' , 'neuromorphometrics', r'C:\Users\Public\Documents\Conda\Lib\site-packages\atlasreader\data\atlases\labels_neuromorphometrics.csv')[1]

print(final_counts_jhu_df.sum(axis=0).sort_values(ascending=False))
print(final_counts_jhutracs_df.sum(axis=0).sort_values(ascending=False))
print(final_counts_oxfcort1mm_df.sum(axis=0).sort_values(ascending=False))
print(final_counts_oxfsubcort1mm_df.sum(axis=0).sort_values(ascending=False))
print(final_counts_neuromorph_df.sum(axis=0).sort_values(ascending=False))
#
#
# Get coordinates of center of anatomical region for atlas 
#
#
coor = defaultdict(list)
def coordinates_atlas(anatomy_img1, n_regions):
    anatomy_img = load_nifti(anatomy_img1)
    stat_img = check_niimg(anatomy_img1)
    aff = stat_img.get_affine()    
    for i in range(0, n_regions):
        region = np.where(anatomy_img==i, 1, 0)
        img=label(region, background=0)
        propsa = regionprops(img)
        coordinates = list(propsa[0].centroid)
        coor[i].append(nib.affines.apply_affine(aff, coordinates))
    return coor
coord = coordinates_atlas(r'Z:\ritter\share\misc\masks\FSL_atlases\JHU\JHU-ICBM-labels-1mm.nii.gz', 49)
coord_df = pd.DataFrame.from_dict(coord, orient='index')
coord_df.columns = ['x']
coord_df = pd.DataFrame(coord_df['x'].tolist(), index=coord_df.index)
coord_df.columns = ['x', 'y', 'z']
labels = pd.read_csv('Z:/home/yelyzaveta/anaconda/lib/python3.5/site-packages/atlasreader/data/atlases/labels_jhuatlas1mm.csv')
coord_df.index = list(labels.iloc[0:49, 1])
#
#
# Get coordinates of lesions centers in xyz cooridnates on image
#
#
def coordinates_image(picture):
        image = load_nifti(picture)
        stat_img = check_niimg(picture)
        aff = stat_img.get_affine()
        img=label(image, background=0)
        propsa1 = regionprops(img)        
        for i in range(len(propsa1)):
            coordinates1[i].append(nib.affines.apply_affine(aff, propsa1[i].centroid))
        return coordinates1
areas_dist_indices = defaultdict(list)
areas_dist_dist = defaultdict(list)
df_baseline_copy = df_baseline.drop([df_baseline.index[62]])
for i in list(df_baseline_copy.iloc[0:98]['path_mask']):
    coordinates1  = defaultdict(list)
    img_coor = coordinates_image(i)
    img_coor_df = pd.DataFrame.from_dict(img_coor, orient='index')
    img_coor_df.columns = ['x']
    img_coor_df = pd.DataFrame(img_coor_df['x'].tolist(), index=img_coor_df.index)
    img_coor_df.columns = ['x', 'y', 'z']
    d = scipy.spatial.distance.cdist(img_coor_df, coord_df, 'cityblock') # get all distances to atlas regions
    min_dist_idx = np.argmin(d, axis=1) #get min distances and sane indices of areas to which is minimal distance 
    min_dist_value = d.min(axis=1) # get min distances and save them 
    areas_dist_indices[i].append(min_dist_idx)
    areas_dist_dist[i].append(min_dist_value)

areas_dist_indices_df = pd.DataFrame.from_dict(areas_dist_indices, orient='index')
areas_dist_dist_df = pd.DataFrame.from_dict(areas_dist_dist, orient='index')
areas_dist_indices_df.columns = ['x']
areas_dist_dist_df.columns = ['x']
areas_dist_indices_df = pd.DataFrame(areas_dist_indices_df['x'].tolist(), index=areas_dist_indices_df.index)
areas_dist_dist_df = pd.DataFrame(areas_dist_dist_df['x'].tolist(), index=areas_dist_dist_df.index)
areas_dist_indices_df.index = [x[63:77] for x in areas_dist_indices_df.index]
areas_dist_dist_df.index = [x[63:77] for x in areas_dist_dist_df.index]
areas_dist_indices_df= areas_dist_indices_df.T
areas_dist_dist_df= areas_dist_dist_df.T
labels['index1'] = labels.index
rename_dict = labels.set_index('index1').to_dict()['name']
areas_dist_indices_df_name = areas_dist_indices_df.replace(rename_dict)
areas_dist_dist_df = areas_dist_dist_df.mask(areas_dist_dist_df.sub(areas_dist_dist_df.mean()).div(areas_dist_dist_df.std()).abs().gt(2))
# 
# 
#  This piece of code I used only to get one table of lesions and their coordinate by one algorithm, by other algotitm and compare them to change in excel file no_lable lesions to label of nearest anatomical region
#
#
# img_coor  = defaultdict(list)
# stat_img = check_niimg(df_baseline_copy.iloc[0]['path_mask'])
# image = load_nifti(df_baseline_copy.iloc[0]['path_mask'])
# aff = stat_img.get_affine()
# img=label(image, background=0)
# propsa1 = regionprops(img)
# coordinat = defaultdict(list)
# for i in range(len(propsa1)):
#            coordinat[i].append(nib.affines.apply_affine(aff, propsa1[i].centroid))
# img_coor_df = pd.DataFrame.from_dict(coordinat, orient='index')
# img_coor_df.columns = ['x']
# img_coor_df = pd.DataFrame(img_coor_df['x'].tolist(), index=img_coor_df.index)
# img_coor_df.columns = ['x', 'y', 'z']
# distances = scipy.spatial.distance.cdist(img_coor_df, coord_df, 'cityblock')
# min_dist_idx = np.argmin(distances, axis=1)
# img_coor_df['dist'] = min_dist_idx
# all_files = glob.glob(os.path.join(r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold', "*_clusters.csv"))
# all_files = sorted(all_files)
# dataframes_peaks = pd.read_csv(all_files[0])
# dataframes_peaks = dataframes_peaks[['peak_x', 'peak_y', 'peak_z']]
# final_counts_jhu_df = make_table_for_specific_lesions(r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold' , 'jhuatlas1mm', r'C:\Users\Public\Documents\Conda\Lib\site-packages\atlasreader\data\atlases\labels_jhuatlas1mm.csv')[2]
# dataframes_peaks['label'] = final_counts_jhu_df.iloc[:, 0]
# merged = dataframes_peaks.join(img_coor_df)
# merged['dist']  = merged['dist'].replace(rename_dict)
# merged[['x', 'y', 'z' ]] = merged[['x', 'y', 'z' ]].round(0)
#
#
#Get the table of names of areas for each patient + calculate the number of areas encountered according to atlas 
#
#
labels_path = r'C:\Users\Public\Documents\Conda\Lib\site-packages\atlasreader\data\atlases\labels_jhuatlas1mm.csv'
dataframe = pd.read_excel (r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\lab\final_counts_jhu_df.xlsx')
dataframe = dataframe.drop(dataframe.columns[[0]], axis=1)
atlas_file = pd.read_csv(labels_path)
atlas_file = atlas_file.drop_duplicates('name')
keyword_dict = {w:w.replace(' ', '_') for w in atlas_file['name']}
final_counts = []
for u in list(dataframe.columns):
    corpus = ' '.join(dataframe[u].fillna(''))
    for w,w2 in keyword_dict.items():
        corpus = corpus.replace(w,w2)    
    all_counts = Counter(corpus.split())
    final_counts.append({w:all_counts[w2] for w,w2 in keyword_dict.items()})
final_counts_df = pd.DataFrame.from_records(final_counts)
all_files = glob.glob(os.path.join(r'C:\Users\Yelyzaveta Snihirova\OneDrive\Рабочий стол\Fold', "*_clusters.csv"))
all_files = sorted(all_files)
names = [x[61:67] for x in all_files]
final_counts_df.index = list(names)
final_counts_df_model = final_counts_df.loc[data_lesions_cut_difference_copy.index]

#
#
#Get models for number of lesions in areas
#
#
model_numbers_anatomical = pd.concat([model_made(final_counts_df_model, data_lesions_cut_difference_copy['edss_score'], 'Number of lesions in areas vs EDSS score', mnb_model),
model_made(final_counts_df_model, data_lesions_cut_difference_copy['manual_t2c_all'], 'Number of lesions in areas vs Number of lesions in 2 years', mnb_model), 
model_made(final_counts_df_model, data_lesions_cut_difference_copy['nondom_hand_average'], 'Number of lesions in areas vs 9HPT score (non-dominant hand)', mnb_model), 
model_made(final_counts_df_model, data_lesions_cut_difference_copy['dom_hand_average'], 'Number of lesions in areas vs 9HPT score (dominant hand)', mnb_model),
model_made(final_counts_df_model, data_lesions_cut_difference_copy['t25wb_average'], 'Number of lesions in areas vs T25FW score', mnb_model)], join='outer', axis=1)

#
#
# Get volumes of lesions in anatomical areas
#
#
def volume_image(picture):
        image = load_nifti(picture)
        img=label(image, background=0)
        props = regionprops(img)        
        for i in range(len(props)):
            coordinat[i].append(props[i].area)
        return coordinat
area_volume = defaultdict(list)
for i in list(df_baseline_copy.iloc[0:98]['path_mask']):
    coordinat  = defaultdict(list)
    volume_imag = volume_image(i)
    area_volume[i].append(volume_imag)
area_volume_df = pd.DataFrame.from_dict(area_volume, orient='index')
area_volume_df.columns = ['x']
area_volume_df = pd.DataFrame(area_volume_df['x'].tolist(), index=area_volume_df.index)
area_volume_df.index = [x[63:77] for x in area_volume_df.index]
area_volume_df= area_volume_df.T
for col in area_volume_df.columns:
    area_volume_df[col] = area_volume_df[col].fillna('[NA]')
area_volume_df = area_volume_df.astype(str)
for col in area_volume_df.columns:
    area_volume_df[col] = area_volume_df[col].str[1:-1]
area_volume_df = area_volume_df.replace('NA', np.nan)
area_volume_df.columns = [str(col) + '_1' for col in area_volume_df.columns]
area_volume_df = area_volume_df.fillna(-1).astype(int)
area_volume_df = area_volume_df.replace(-1, np.nan)
areas_volume_where = pd.concat([area_volume_df, areas_dist_indices_df_name], axis=1, join="outer")
areas_volume_where = areas_volume_where.reindex(sorted(areas_volume_where.columns), axis=1)
#
#
# Get table of sum of pixels for lesions for specific ares for patients
#
#
volume_lesions = []
column_numb = 0
while column_numb<=195 and len(volume_lesions)<= 98:
    for i in areas_volume_where.columns[[column_numb]]:
        for j in areas_volume_where.columns[[column_numb+1]]:
            f = pd.DataFrame({i: areas_volume_where[i],
                j: areas_volume_where[j]})
            b = f.groupby(i).agg({j: [sum]})
            volume_lesions.append(b)
            column_numb = column_numb+2
volume_lesions_final = pd.concat(volume_lesions, join='outer', axis=1)
volume_lesions_final.columns = areas_dist_indices_df_name.columns
volume_lesions_final['0094B3_sess_01'] = np.nan
volume_lesions_final= volume_lesions_final.reindex(sorted(volume_lesions_final.columns), axis=1)
for col in volume_lesions_final.columns:
    volume_lesions_final[col] = volume_lesions_final[col].fillna(0)
volume_lesions_final= volume_lesions_final.T
volume_lesions_final.index = [x[0:6] for x in volume_lesions_final.index]
volume_lesions_final = volume_lesions_final.loc[data_lesions_cut_difference_copy.index]
#
#
# Build models for volumes of lesions in anatomical areas
#
#

model_volumes_anatomical =pd.concat([model_made(volume_lesions_final, data_lesions_cut_difference_copy['edss_score'], 'Volume of lesions in areas vs EDSS score', gnb_model),
model_made(volume_lesions_final, data_lesions_cut_difference_copy['manual_t2c_all'], 'Volume of lesions in areas vs Number of lesions in 2 years', gnb_model),
model_made(volume_lesions_final, data_lesions_cut_difference_copy['nondom_hand_average'], 'Volume of lesions in areas vs 9HPT score (non-dominant hand)', gnb_model),
model_made(volume_lesions_final, data_lesions_cut_difference_copy['dom_hand_average'], 'Volume of lesions in areas vs 9HPT score (dominant hand)', gnb_model),
model_made(volume_lesions_final, data_lesions_cut_difference_copy['t25wb_average'], 'Volume of lesions in areas vs T25FW score', gnb_model)], join='outer', axis=1)
