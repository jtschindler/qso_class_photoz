import numpy as np
import pandas as pd
import math

import ml_quasar_sample as qs
import ml_sets as sets
import rf_class as rf_class

def test_all():

    label = "label"

    #SDSS+2MASS+WISE2
    print "SDSS+2MASS+WISE2 w/o SDSS colors"

    df_stars = pd.read_csv('../DR10_star_flux_cat.csv')
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3', \
            # 'WISE_w4', \
            ]

    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df = qs.build_full_sample(df_stars, df_quasars, 10)

    labels = ["STAR","QSO"]
    features = ['WISE_w1','2MASS_j','hj', 'jks', 'ksw1', 'w1w2']

    print 'Number of Quasars: ', df.query('label =="QSO"').shape[0]
    print 'Number of Stars: ', df.query('label =="STAR"').shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [25,50,100], 'min_samples_split': [2,3,4],
                    'max_depth' : [10,15,20]}]

    rf_class.rf_class_grid_search(df,features, label, param_grid)

    print"\n"
    print"\n"
    print"\n"


    #SDSS+2MASS+WISE3
    print "SDSS+2MASS+WISE3 w/o SDSS colors"

    df_stars = pd.read_csv('../DR10_star_flux_cat.csv')
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            'WISE_w3', \
            # 'WISE_w4', \
            ]

    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df = qs.build_full_sample(df_stars, df_quasars, 10)

    labels = ["STAR","QSO"]
    features = ['WISE_w1','2MASS_j','hj', 'jks', 'ksw1', 'w1w2','w2w3']

    print 'Number of Quasars: ', df.query('label =="QSO"').shape[0]
    print 'Number of Stars: ', df.query('label =="STAR"').shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [25,50,100], 'min_samples_split': [2,3,4],
                    'max_depth' : [10,15,20]}]

    rf_class.rf_class_grid_search(df,features, label, param_grid)

    print"\n"
    print"\n"
    print"\n"

    #SDSS+2MASS+WISE2
    print "SDSS+2MASS+WISE2"

    df_stars = pd.read_csv('../DR10_star_flux_cat.csv')
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3', \
            # 'WISE_w4', \
            ]

    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df = qs.build_full_sample(df_stars, df_quasars, 10)

    labels = ["STAR","QSO"]
    features = ['SDSS_i','WISE_w1','2MASS_j','ug','gr','ri','iz','zh','hj', 'jks', 'ksw1', 'w1w2']

    print 'Number of Quasars: ', df.query('label =="QSO"').shape[0]
    print 'Number of Stars: ', df.query('label =="STAR"').shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [25,50,100], 'min_samples_split': [2,3,4],
                    'max_depth' : [10,15,20]}]

    rf_class.rf_class_grid_search(df,features, label, param_grid)

    print"\n"
    print"\n"
    print"\n"


    #SDSS+2MASS+WISE3
    print "SDSS+2MASS+WISE3 SDSS colors"

    df_stars = pd.read_csv('../DR10_star_flux_cat.csv')
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            'WISE_w3', \
            # 'WISE_w4', \
            ]

    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df = qs.build_full_sample(df_stars, df_quasars, 10)

    labels = ["STAR","QSO"]
    features = ['SDSS_i','WISE_w1','2MASS_j','ug','gr','ri','iz','zh','hj', 'jks', 'ksw1', 'w1w2','w2w3']

    print 'Number of Quasars: ', df.query('label =="QSO"').shape[0]
    print 'Number of Stars: ', df.query('label =="STAR"').shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [25,50,100], 'min_samples_split': [2,3,4],
                    'max_depth' : [10,15,20]}]

    rf_class.rf_class_grid_search(df,features, label, param_grid)

    print"\n"
    print"\n"
    print"\n"

    #SDSS
    print "SDSS"

    df_stars = pd.read_csv('../DR10_star_flux_cat.csv')
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # '2MASS_h','2MASS_j','2MASS_ks', \
            # 'WISE_w1','WISE_w2', \
            # 'WISE_w3', \
            # 'WISE_w4', \
            ]

    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df = qs.build_full_sample(df_stars, df_quasars, 10)

    labels = ["STAR","QSO"]
    features = ['SDSS_i','ug','gr','ri','iz']

    print 'Number of Quasars: ', df.query('label =="QSO"').shape[0]
    print 'Number of Stars: ', df.query('label =="STAR"').shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [25,50,100], 'min_samples_split': [2,3,4],
                    'max_depth' : [10,15,20]}]

    rf_class.rf_class_grid_search(df,features, label, param_grid)

    print"\n"
    print"\n"
    print"\n"


    #SDSS+2MASS+WISE4
    print "SDSS+2MASS+WISE4 w/o SDSS colors"

    df_stars = pd.read_csv('../DR10_star_flux_cat.csv')
    df_quasars = pd.read_csv('../DR7_DR12Q_flux_cat.csv')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            '2MASS_h','2MASS_j','2MASS_ks', \
            'WISE_w1','WISE_w2', \
            'WISE_w3', \
            'WISE_w4', \
            ]

    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df = qs.build_full_sample(df_stars, df_quasars, 10)

    labels = ["STAR","QSO"]
    features = ['WISE_w1','2MASS_j','hj', 'jks', 'ksw1', 'w1w2','w2w3','w3w4']

    print 'Number of Quasars: ', df.query('label =="QSO"').shape[0]
    print 'Number of Stars: ', df.query('label =="STAR"').shape[0]
    print passband_names
    print features

    param_grid = [{'n_estimators': [25,50,100], 'min_samples_split': [2,3,4],
                    'max_depth' : [10,15,20]}]

    rf_class.rf_class_grid_search(df,features, label, param_grid)

    print"\n"
    print"\n"
    print"\n"


"""
MAIN ROUTINE
"""


df_stars = pd.read_csv('models/DR10_star_flux_cat.csv')
df_quasars = pd.read_csv('models/DR7_DR12Q_flux_cat.csv')

passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
        '2MASS_j', \
        '2MASS_h', \
        '2MASS_ks', \
        'WISE_w1','WISE_w2', \
        # 'WISE_w3', \
        # 'WISE_w4', \
        ]

print df_stars.shape
print df_quasars.shape

df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)
print features
print df_stars.shape
print df_quasars.shape

# TODO: Evaluation routine, SVM grid search, ...
# TODO : Separate files for SVM and RF methods, new testing file , put sets and
#  sample file together, check imports

#Reduce the total set of objects for testing the routines
df_stars = df_stars.sample(frac=1.0)
df_quasars = df_quasars.sample(frac=1.0)

# print df_stars.columns
# print df_quasars.columns

#Impose allsky selection criteria on the dataframes
# df_stars.query('psfmag_I',inplace=True)
# df_stars.query(' (K_M_2MASS - W2MPRO) >= -0.848 * (J_M_2MASS - K_M_2MASS) + 1.8 ',inplace=True)
# df_quasars.query(' (K_M_2MASS - W2MPRO) >= -0.848 * (J_M_2MASS - K_M_2MASS) + 1.8 ',inplace=True)
# df_stars.query('PSFMAG_I <= 18.5',inplace=True)
# df_quasars.query('PSFMAG_I <= 18.5',inplace=True)
# print df_stars.shape
# print df_quasars.shape

df = qs.build_full_sample(df_stars, df_quasars, 50)
# df_quasars['label'] = 'QSO'
# df_stars['label'] = 'STAR'
# df = pd.concat([df_quasars,df_stars])

labels = ["STAR","QSO"]
features = ['WISE_w1','2MASS_j', 'jh', 'hks', 'ksw1', 'w1w2']
# features = ['SDSS_i','WISE_w1','2MASS_j','ug','gr','ri','iz','zh','hj', 'jks', 'ksw1', 'w1w2','w2w3']
# features = ['SDSS_i','ug','gr','ri','iz']

label = 'label'

params = {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 3,
    'n_jobs': 4, 'random_state': 0}
# rf_class.rf_class_example(df, features, label, params)

param_grid = [{'n_estimators': [25,50,100,200], 'min_samples_split': [2,3,4],
                'max_depth' : [10,15,20]}]

rf_class.rf_class_grid_search(df,features, label ,param_grid)

# test_all()

# param_range = [1,2,3,4,5,6,7,8,9,10]
# rf_class.rf_class_validation_curve(df,features,label, "n_estimators",param_range)
# pl.show_features(df,features,labels,0.1)


# rf_class.rf_class_roc_curve(df,features,label)
