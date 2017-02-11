import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from sklearn import preprocessing, cross_validation

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


from class_photoz import ml_sets as sets
from class_photoz import ml_quasar_sample as qs
from class_photoz import rf_reg as rf
from class_photoz import svm_reg as svr
from class_photoz import ml_analysis as ml_an
from class_photoz import photoz_analysis as pz_an


def DR7DR12_grid_search():
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
    param_grid = [{'C': [10,1.0,0.1], 'gamma': [0.01,0.1,1.0], \
                'kernel': ['rbf'],'epsilon':[0.1,0.2,0.3]}]

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

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5W1W2')

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
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5a')

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

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5b')

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
    df_train.query('PSFMAG_I < 18.5',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5W1W2_icut')

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
    df_train.query('PSFMAG_I < 18.5',inplace=True)
    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

       # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'DR7DR12_SDSS5b_icut')

    


def simqsos_grid_search():
    # --------------------------------------------------------------------------
    # Read data file and input parameters
    # --------------------------------------------------------------------------

    df = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')

    df = df.sample(frac=0.1)



    label = 'z'
    rand_state = 1
    param_grid = [{'C': [10], 'gamma': [0.1], \
                'kernel': ['rbf']}]
    # scores = ['neg_mean_absolute_error','neg_mean_squared_error','r2',]
    scores = ['neg_mean_absolute_error']
    params = {'cv':5,'n_jobs':2}

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

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'simqsos_SDSS5W1W2')

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

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'simqsos_SDSS5a')

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

    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)

    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','ug','gr','ri','iz']

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'simqsos_SDSS5b')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------



def test_example():
    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    # df_train = pd.read_hdf('../class_photoz/data/DR7DR14Q_clean_flux_cat.hdf5','data')
    df_train = pd.read_hdf('../class_photoz/data/brightqsos_deep.hdf5','data')
    # Try a fraction of the whole datafile first
    df_train = df_train.sample(frac=0.5)

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

    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    label = 'z'
    rand_state = 1

    params = {'kernel':'rbf','C':10, 'gamma':0.1, 'epsilon':0.1,'cache_size':1200}


    svr.svm_reg_example(df_train,features,label,params,rand_state)

def predict_example():

    # UNRESOLVED ISSUES WITH PREDICTION

    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_test = pd.read_hdf('../class_photoz/data/DR7DR14Q_clean_flux_cat.hdf5','data')
    # df_train = pd.read_hdf('../class_photoz/data/DR7DR14Q_flux_cat.hdf5','data')
    df_train = pd.read_hdf('../class_photoz/data/brightqsos_try.hdf5','data')
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    # Try a fraction of the whole datafile first
    df_train = df_train.sample(frac=0.2)

    # df_test.query('Z > 1.1',inplace=True)
    # df_train.query('z > 1.1',inplace=True)
    df_test.query('2.0 > Z_VI > 0.2 ',inplace=True)
    df_train.query('2.0 > z > 0.2',inplace=True)

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

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    # features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh', 'hk', 'kw1', 'w1w2']
    label = 'z'


    params = {'kernel':'rbf', 'C':1.0, 'gamma':0.001, 'epsilon':0.2, 'cache_size':1200}

    df_test = svr.svm_reg_predict(df_train, df_test, features, label, params, 'svm_photoz')
    print df_test['svm_photoz'].describe()
    ml_an.evaluate_regression(df_test['Z_VI'],df_test['svm_photoz'])
    pz_an.plot_redshifts(df_test['Z_VI'],df_test['svm_photoz'])
    pz_an.plot_error_hist(df_test['Z_VI'],df_test['svm_photoz'])
    plt.show()



#test_example()
# simqsos_grid_search()
DR7DR12_grid_search()
# grid_search_example()
# predict_example()
