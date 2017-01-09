import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

import ml_quasar_sample as qs
import ml_analysis as ml_an

import ml_sets as sets
import eval_quasarstar_photoz_pred as ev

import rf_reg as rf_reg
import rf_class as rf_class

# -----------------------------------------------------------------------------
# Include the Training data
# -----------------------------------------------------------------------------

df_stars = pd.read_csv('models/DR10_star_flux_cat.csv')
df_quasars = pd.read_csv('models/DR7_DR12Q_flux_cat.csv')
# df_stars.query(' (K_M_2MASS - W2MPRO) >= -0.848 * (J_M_2MASS - K_M_2MASS) + 1.8 ',inplace=True)
# df_quasars.query(' (K_M_2MASS - W2MPRO) >= -0.848 * (J_M_2MASS - K_M_2MASS) + 1.8 ',inplace=True)
df_to_predict = pd.read_hdf('models/wise_tmass_sdss_bright_fluxes.hdf5','data')

print df_quasars.shape

# Excluding objects in allsky selection from training as much as possible
# df_quasars.loc[:,'new_designation'] = 'SDSS J'+(df_quasars.loc[:,'designation'])
# df_quasars.loc[:,'excl'] = df_quasars.isin(df_to_predict.milliquas_name.values).new_designation
# df_quasars.drop(df_quasars[df_quasars['excl']].index.tolist(),inplace=True)
# print df_quasars.shape
# df_quasars.loc[:,'new_designation'] = 'J'+(df_quasars.loc[:,'designation'])
# df_quasars.loc[:,'excl'] = df_quasars.isin(df_to_predict.wise_designation.values).new_designation
# df_quasars.drop(df_quasars[df_quasars['excl']].index.tolist(),inplace=True)
# print df_quasars.shape

#Cheating to use unwise instead of WISE colors
# df_to_predict['WISE_w1'] = df_to_predict['UNWISE_w1']
# df_to_predict['WISE_w2'] = df_to_predict['UNWISE_w2']


# -----------------------------------------------------------------------------
# Quasar-Star-Classification
# -----------------------------------------------------------------------------


passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
        '2MASS_j','2MASS_h','2MASS_ks', \
        'WISE_w1','WISE_w2', \
        # 'WISE_w3', \
        # 'WISE_w4', \
        ]

df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)
df_to_predict,features = qs.prepare_flux_ratio_catalog(df_to_predict,passband_names)
print "QUASARS in classification training sample : ",df_quasars.shape[0]
df = qs.build_full_sample(df_stars, df_quasars, 50)

print df.query('label=="QSO"').shape[0]
print df.query('label=="STAR"').shape[0]

labels = ["STAR","QSO"]
features = ['WISE_w1','2MASS_j','jh', 'hks', 'ksw1', 'w1w2']

params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 3,
    'n_jobs': 4, 'random_state': 0}


df_to_predict = rf_class.rf_class_predict(df, df_to_predict, features, params,
    'label', 'rf_class_pred','rf_qso_prob','rf_star_prob')

# df_to_predict['rf_qso_prob'] = df_to_predict2['rf_qso_prob']
# df_to_predict['rf_star_prob'] = df_to_predict2['rf_star_prob']
# df_to_predict['rf_class_pred'] = df_to_predict2['rf_class_pred']

# print df_to_predict.rf_class_pred.value_counts()
# print df_to_predict.columns
# ev.eval_quasar_star_pred(df_to_predict)

# -----------------------------------------------------------------------------
# Photometric redshift estimation
# -----------------------------------------------------------------------------


# Load the Quasar catalog
df_quasars = pd.read_csv('models/DR7_DR12Q_flux_cat.csv')
df_to_predict2 = pd.read_hdf('models/wise_tmass_sdss_bright_fluxes.hdf5','data')

#Cheating to use unwise instead of WISE colors
# df_to_predict2['WISE_w1'] = df_to_predict2['UNWISE_w1']
# df_to_predict2['WISE_w2'] = df_to_predict2['UNWISE_w2']

# Set passband names for flux ratio calculation
passband_names = [\
        'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
        # '2MASS_h','2MASS_j','2MASS_ks', \
        'WISE_w1','WISE_w2', \
        # 'WISE_w3' \
        ]

# Calculate flux ratios and save all flux ratios as features
df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names,sigma=True)

df_to_predict2,features = qs.prepare_flux_ratio_catalog(df_to_predict2,passband_names)

# df_quasars.to_csv('temp.csv')

features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

# Set the label for regression
label = 'redshift'

# Set parameters for random forest regression
params = {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 3, 'n_jobs': 4}

df_to_predict2 = rf_reg.rf_reg_predict(df_quasars, df_to_predict2, features, label, params, 'rf_photoz')

print df_to_predict.shape
print df_to_predict2.shape

df_to_predict['rf_photoz'] = df_to_predict2['rf_photoz']


df_to_predict.to_hdf('wise_tmass_sdss_bright_pred.hdf5','data')

# df_to_predict = pd.read_csv('cat_with_pred.csv')

print "\n"
print "Shape of predicted array is now : ", df_to_predict.shape
print "\n"
ev.eval_quasar_star_pred(df_to_predict)
print "\n"
print "Shape of predicted array is now : ", df_to_predict.shape
print "\n"
ev.eval_photoz_pred(df_to_predict)
print "\n"
print "Shape of predicted array is now : ", df_to_predict.shape
print "\n"
cat = ev.calc_priorities(df_to_predict)
print "\n"
print "Shape of predicted array is now : ", df_to_predict.shape
print "\n"
ev.eval_observed_cand(cat)
print "\n"
print "Shape of predicted array is now : ", df_to_predict.shape
print "\n"
ev.select_qso_candidates(cat)
print "\n"
print "Shape of predicted array is now : ", df_to_predict.shape

# ev.eval_obs_obj(df_to_predict)
