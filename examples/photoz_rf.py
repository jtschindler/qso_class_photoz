import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from sklearn import preprocessing


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from class_photoz import ml_sets as sets
from class_photoz import ml_quasar_sample as qs
from class_photoz import rf_reg as rf
from class_photoz import ml_analysis as ml_an
from class_photoz import photoz_analysis as pz_an

def grid_search_example():

    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_train = pd.read_hdf('../class_photoz/data/DR7DR14Q_flux_cat.hdf5','data')

    df_train = df_train.sample(frac=0.1)

    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    df_train.replace(np.inf, np.nan,inplace=True)

    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    label = 'Z'
    rand_state = 1
    param_grid = [{'n_estimators': [5,10], 'min_samples_split': [2], \
                     'max_depth' : [15]} ]
    # scores = ['neg_mean_absolute_error','neg_mean_squared_error','r2',]
    scores = ['r2',]



    rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'example')

def test_example():
    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_train = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    # df_train = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k.hdf5','data')

    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    # for name in passband_names:
    #     df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
    #     df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)

    df_train.replace(np.inf, np.nan,inplace=True)
    df_train.query('10 > Z_VI > 0.0 and PSFMAG_I < 18.5',inplace=True)
    # df_train.query('obsMag_SDSS_i < 19.0',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # df_train = df_train.sample(frac=0.5)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    # features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh', 'hk', 'kw1', 'w1w2']
    label = 'z'
    rand_state = 1

    params = {'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 2, 'n_jobs': 2, 'random_state':rand_state,}


    rf.rf_reg_example(df_train,features,label,params,rand_state,save=True,save_filename='test')


def predict_example():
    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_test = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    df_train = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k.hdf5','data')
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    df_test.query('5.5 > Z_VI > 0.2 and 18.5 > PSFMAG_I',inplace=True)
    df_train.query('z > 0.2 and obsMag_SDSS_i < 18.5',inplace=True)

    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)


    df_test.replace(np.inf, np.nan,inplace=True)
    df_train.replace(np.inf, np.nan,inplace=True)

    df_test,features = qs.prepare_flux_ratio_catalog(df_test,passband_names)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    print df_test.shape, df_train.shape
    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    # features = ['SDSS_i','ug','gr','ri','iz']
    # features = ['SDSS_u','SDSS_i','SDSS_r','SDSS_z','SDSS_g','WISE_w1','WISE_w2',\
                # 'sigma_SDSS_u','sigma_SDSS_g','sigma_SDSS_r','sigma_SDSS_i','sigma_SDSS_z',\
                # 'sigma_WISE_w1','sigma_WISE_w2']
    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    # features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2','w2w3']
    # features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2',\
                # 'sigma_SDSS_u','sigma_SDSS_g','sigma_SDSS_r','sigma_SDSS_i',\
                # 'sigma_SDSS_z','sigma_WISE_w1','sigma_WISE_w2',\
                # ]
    # features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh', 'hk', 'kw1', 'w1w2']

    label = 'z'
    rand_state = 1

    params = {'n_estimators':200, 'max_depth': 25, 'min_samples_split': 3, 'n_jobs': 4, 'random_state':rand_state}

    df_test = rf.rf_reg_predict(df_train, df_test, features, label, params, 'rf_photoz')

    ml_an.evaluate_regression(df_test['Z_VI'],df_test['rf_photoz'])
    pz_an.plot_redshifts(df_test['Z_VI'],df_test['rf_photoz'])
    pz_an.plot_error_hist(df_test['Z_VI'],df_test['rf_photoz'])
    plt.show()

# grid_search_example()

test_example()

# predict_example()
