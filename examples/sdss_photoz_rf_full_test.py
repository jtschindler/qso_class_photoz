
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from class_photoz import ml_quasar_sample as qs
from class_photoz import rf_reg as rf
from class_photoz import ml_analysis as ml_an
from class_photoz import photoz_analysis as pz_an

import class_photoz as cl


def dr7dr12_grid_search():
    # --------------------------------------------------------------------------
    # Read data file and input parameters
    # --------------------------------------------------------------------------
    df = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')

    df = df.query('0 < Z_VI < 10')

    df.replace(np.inf, np.nan,inplace=True)

    # scores = ['neg_mean_absolute_error','neg_mean_squared_error','r2',]
    scores = ['r2']

    label = 'Z_VI'
    rand_state = 1
    param_grid = [{'n_estimators': [50,100,200,300], 'min_samples_split': [2,3,4], \
                     'max_depth' : [15,20,25]} ]

    # param_grid = [{'n_estimators': [200], 'min_samples_split': [4], \
    #                  'max_depth' : [15]} ]

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)
    # df_train = df_train.sample(frac=0.5)
    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    # rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5W1W2')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    # rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5a')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            # 'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    # rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5b')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)
    df_train.query('SDSS_mag_i <= 18.5',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5W1W2_icut')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            # 'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)
    df_train.query('SDSS_mag_i <= 18.5',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5b_icut')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)
    df_train.query('SDSS_mag_i <= 18.5',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5a_icut')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------


def simqsos_grid_search():
    # --------------------------------------------------------------------------
    # Read data file and input parameters
    # --------------------------------------------------------------------------

    df = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')

    # df = df.sample(frac=0.1)



    label = 'z'
    rand_state = 1
    param_grid = [{'n_estimators': [50,100,200,300], 'min_samples_split': [2,3,4], \
                     'max_depth' : [15,20,25]} ]
    # scores = ['neg_mean_absolute_error','neg_mean_squared_error','r2',]
    scores = ['r2']


    df.replace(np.inf, np.nan,inplace=True)

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    df_train = df.copy(deep=True)

    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)

    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    # rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'simqsos_SDSS5W1W2')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    df_train = df.copy(deep=True)

    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)

    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    # rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'simqsos_SDSS5a')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)

    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)

    df_train.query('obsMag_SDSS_i <= 18.5',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    # rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'simqsos_SDSS5W1W2_icut')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)

    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)

    df_train.query('obsMag_SDSS_i <= 18.5',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    # rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'simqsos_SDSS5_icut')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------


    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    df_train = df.copy(deep=True)

    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)

    df_train.query('obsMag_SDSS_i <= 18.5',inplace=True)
    df_train['kw2'] = df_train.obsMag_TMASS_k-df_train.obsMag_WISE_w2
    df_train['jk'] = df_train.obsMag_TMASS_j-df_train.obsMag_TMASS_k
    df_train.query('kw2 >= -0.501208-0.848*jk',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    rf.rf_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'simqsos_SDSS5W1W2_icut_colorcut')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------


def simqso_test():
    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------

    df_train = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)


    df_train.replace(np.inf, np.nan,inplace=True)
    # df_train.query('obsMag_SDSS_i <= 18.5',inplace=True)


    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    # features = ['SDSS_i','ug','gr','ri','iz']

    label = 'z'
    rand_state = 1

    params = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 4, 'n_jobs': 2, 'random_state':rand_state}


    rf.rf_reg_example(df_train,features,label,params,rand_state,save=True,save_filename='rf_sim_sdssw1w2')







def dr7dr12_test():

    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_train = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')

    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]


    df_train.replace(np.inf, np.nan,inplace=True)
    df_train = df_train.query('0 < Z_VI < 10')

    # df_train.query('SDSS_mag_i <= 18.5',inplace=True)



    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    # features = ['SDSS_i','ug','gr','ri','iz']
    label = 'Z_VI'
    rand_state = 1

    params = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2, 'n_jobs': 2, 'random_state':rand_state}

    print df_train.shape[0]

    rf.rf_reg_example(df_train,features,label,params,rand_state,save=True,save_filename='rf_sdssw1w2')




def simqso_predict_dr7dr12():
    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_test = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    df_train = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]


    df_test = df_test.query('0.3 < Z_VI < 5.5')
    # df_train.query('obsMag_SDSS_i <= 18.5',inplace=True)
    # df_test.query('SDSS_mag_i <= 18.5',inplace=True)
    # df_train.query('z > 1.1',inplace=True)

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

    # features = ['SDSS_u','SDSS_i','SDSS_r','SDSS_z','SDSS_g','WISE_w1','WISE_w2']
    #  features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    features = ['ug','gr','ri','iz','zw1','w1w2']
    # features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh', 'hks', 'ksw1', 'w1w2']
    label = 'z'
    rand_state = 1

    params = {'n_estimators': 300, 'max_depth': 30, 'min_samples_split': 4, 'n_jobs': 2, 'random_state':rand_state}

    df_test = rf.rf_reg_predict(df_train, df_test, features, label, params, 'rf_photoz')

    ml_an.evaluate_regression(df_test['Z_VI'],df_test['rf_photoz'])
    pz_an.plot_redshifts(df_test['Z_VI'],df_test['rf_photoz'])
    pz_an.plot_error_hist(df_test['Z_VI'],df_test['rf_photoz'])
    plt.show()




def dr7dr12_predict_simqso():
    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_train = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    df_test = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]


    df_train = df_train.query('0.3 < Z_VI < 5.5')
    df_test.query('obsMag_SDSS_i <= 18.5',inplace=True)
    df_train.query('SDSS_mag_i <= 18.5',inplace=True)
    # df_train.query('z > 1.1',inplace=True)

    for name in passband_names:
        df_test.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_test.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)


    df_test.replace(np.inf, np.nan,inplace=True)
    df_train.replace(np.inf, np.nan,inplace=True)

    df_test,features = qs.prepare_flux_ratio_catalog(df_test,passband_names)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    print df_test.shape, df_train.shape
    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    # features = ['SDSS_u','SDSS_i','SDSS_r','SDSS_z','SDSS_g','WISE_w1','WISE_w2']
    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    # features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh', 'hks', 'ksw1', 'w1w2']
    label = 'Z_VI'
    rand_state = 1

    params = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 2, 'n_jobs': 2, 'random_state':rand_state}

    df_test = rf.rf_reg_predict(df_train, df_test, features, label, params, 'rf_photoz')

    ml_an.evaluate_regression(df_test['z'],df_test['rf_photoz'])
    pz_an.plot_redshifts(df_test['z'],df_test['rf_photoz'])
    pz_an.plot_error_hist(df_test['z'],df_test['rf_photoz'])
    plt.show()




# DR7DR12_grid_search()
# test_example()
# dr7dr12_test()
# simqso_test()
# simqsos_grid_search()
# simqso_predict_dr7dr12()
# dr7dr12_predict_simqso()
