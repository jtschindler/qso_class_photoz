import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from sklearn import preprocessing, cross_validation

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split


import ml_sets as sets
import ml_quasar_sample as qs
import rf_reg as rf



def full_test():

    # 1) SDSS+WISE3
    print "SDSS+WISE3\n"
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            'WISE_w3' \
            ]

    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2','w2w3']
    label = 'redshift'

    print 'Number of Quasars: ', df_quasars.shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [5,10,50,100], 'min_samples_split': [2,3,4], \
                     'max_depth' : [15,20,25]} ]

    rf.rf_reg_grid_search(df_quasars,features,label,param_grid)

    print"\n"
    print"\n"
    print"\n"

    # 2) SDSS+WISE2
    print "SDSS+WISE2\n"
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]


    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    label = 'redshift'

    print 'Number of Quasars: ', df_quasars.shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [5,10,50,100], 'min_samples_split': [2,3,4], \
                     'max_depth' : [15,20,25]} ]

    rf.rf_reg_grid_search(df_quasars,features,label,param_grid)

    print"\n"
    print"\n"
    print"\n"

    # 3) SDSS+2MASS+WISE2
    print "SDSS+2MASS+WISE2\n"
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]


    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    features = ['SDSS_i','WISE_w1','2MASS_j','ug','gr','ri','iz','zh','hj',
    'jks', 'ksw1', 'w1w2']
    label = 'redshift'

    print 'Number of Quasars: ', df_quasars.shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [5,10,50,100], 'min_samples_split': [2,3,4], \
                     'max_depth' : [15,20,25]} ]

    rf.rf_reg_grid_search(df_quasars,features,label,param_grid)

    print"\n"
    print"\n"
    print"\n"

    # 4) SDSS
    print "SDSS\n"
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # '2MASS_h','2MASS_j','2MASS_ks', \
            # 'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]


    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    features = ['SDSS_i','ug','gr','ri','iz']
    label = 'redshift'

    print 'Number of Quasars: ', df_quasars.shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [5,10,50,100], 'min_samples_split': [2,3,4], \
                     'max_depth' : [15,20,25]} ]

    rf.rf_reg_grid_search(df_quasars,features,label,param_grid)

    print"\n"
    print"\n"
    print"\n"

    # 5) SDSS+2MASS
    print "SDSS+2MASS\n"
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            '2MASS_h','2MASS_j','2MASS_ks', \
            # 'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]


    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    features = ['SDSS_i','2MASS_j','ug','gr','ri','iz','zh','hj','jks']
    label = 'redshift'

    print 'Number of Quasars: ', df_quasars.shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [5,10,50,100], 'min_samples_split': [2,3,4], \
                     'max_depth' : [15,20,25]} ]

    rf.rf_reg_grid_search(df_quasars,features,label,param_grid)

    print"\n"
    print"\n"
    print"\n"

    # 6) SDSS+2MASS+WISE2
    print "SDSS+2MASS+WISE w/o SDSS colors\n"
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    features = ['WISE_w1','2MASS_j','hj', 'jks', 'ksw1', 'w1w2']
    label = 'redshift'

    print 'Number of Quasars: ', df_quasars.shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [5,10,50,100], 'min_samples_split': [2,3,4], \
                     'max_depth' : [15,20,25]} ]

    rf.rf_reg_grid_search(df_quasars,features,label,param_grid)



"""
MAIN ROUTINE
"""
# -----------------------------------------------------------------------------
# DATA PREPARATION
# -----------------------------------------------------------------------------

# Load the Quasar catalog
df_quasars = pd.read_csv('../class_photoz/data/DR7_DR12Q_flux_cat.csv')
# Set passband names for flux ratio calculation
passband_names = [\
        'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
        # '2MASS_h','2MASS_j','2MASS_ks', \
        'WISE_w1','WISE_w2', \
        # 'WISE_w3' \
        ]

# Calculate flux ratios and save all flux ratios as features
df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

# Manually set the list of features
# features = ['SDSS_i','ug','gr','ri','iz', \
# 'sigma_gr','sigma_ri','sigma_iz','weight']
# features = ['SDSS_i','ug','gr','ri','iz','zh','WISE_w1','2MASS_j','hj', 'jks', 'ksw1', 'w1w2','w2w3']
features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
# 'sigma_gr','sigma_ri','sigma_iz','sigma_zw1','sigma_w1w2']
# features = ['SDSS_i','2MASS_j','ug','gr','ri','iz','zh','hj', 'jks']
# features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1', 'w1w2']
# features = ['SDSS_i','WISE_w1','2MASS_j','ug','gr','ri','iz','zh','hj', 'jks', 'ksw1', 'w1w2']
# features = ['WISE_w1','2MASS_j','hj', 'jks', 'ksw1', 'w1w2']

# Set the label for regression
label = 'redshift'

# -----------------------------------------------------------------------------
# RANDOM FOREST REGRESSION
# -----------------------------------------------------------------------------

# Set parameters for random forest regression
params = {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 2, 'n_jobs': 4}


# Run the example
rf.rf_reg_example(df_quasars,features,label,params)

# Set parameters for random forest grid search
# param_grid = [{'n_estimators': [25,50,100,200], 'min_samples_split': [2,3,4], \
                # 'max_depth' : [15,20,25]} ]
# param_grid = [{'n_estimators': [25,50], 'min_samples_split': [2,3,4], \
                # 'max_depth' : [15]} ]

# Perform random forest grid search
# rf.rf_reg_grid_search(df_quasars,features,label,param_grid)

# Set parameters for validation curve
# param_name = "n_estimators"
# param_range = [10,50,100,200]
# param_name = 'max_depth'
# param_range = [5,10,15,20,25]

# Perform validation curve calculations
# rf.rf_reg_validation_curve(df_quasars,features,label, params, param_name,param_range)


# full_test()
