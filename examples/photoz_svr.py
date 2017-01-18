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


from class_photoz import ml_sets as sets
from class_photoz import ml_quasar_sample as qs
from class_photoz import rf_reg as rf
from class_photoz import svm_reg as svr
from class_photoz import ml_analysis as ml_an
from class_photoz import photoz_analysis as pz_an


def grid_search_example():

    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_train = pd.read_hdf('../class_photoz/data/DR7DR14Q_flux_cat.hdf5','data')

    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    # Try a fraction of the whole datafile first
    df_train = df_train.sample(frac=0.05)

    df_train.replace(np.inf, np.nan,inplace=True)

    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    label = 'Z'
    rand_state = 1
    param_grid = [{'C': [1, 10], 'gamma': [0.0001], \
                'kernel': ('rbf','linear')}]
    scores = ['mean_absolute_error','mean_squared_error','r2',]

    svr.svm_reg_grid_search(df_train,features,label,param_grid,rand_state,scores,'example')


def test_example():
    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_train = pd.read_hdf('../class_photoz/data/DR7DR14Q_flux_cat.hdf5','data')

    # Try a fraction of the whole datafile first
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

    params = {'kernel':'linear','C':1.0, 'gamma':0.001, 'epsilon':0.2,'cache_size':1200}


    svr.svm_reg_example(df_train,features,label,params,rand_state)

def predict_example():

    # UNRESOLVED ISSUES WITH PREDICTION

    # --------------------------------------------------------------------------
    # Preparing the feature matrix
    # --------------------------------------------------------------------------
    df_test = pd.read_hdf('../class_photoz/data/DR7DR14Q_flux_cat.hdf5','data')
    # df_train = pd.read_hdf('../class_photoz/data/DR7DR14Q_flux_cat.hdf5','data')
    df_train = pd.read_hdf('../class_photoz/data/brightqsos_2.hdf5','data')
    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_ks', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]
    # Try a fraction of the whole datafile first
    df_train = df_train.sample(frac=1.0)

    # df_test.query('Z > 1.1',inplace=True)
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

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    # features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh', 'hks', 'ksw1', 'w1w2']
    label = 'z'


    params = {'kernel':'rbf', 'C':1.0, 'gamma':0.001, 'epsilon':0.2, 'cache_size':1200}

    df_test = svr.svm_reg_predict(df_train, df_test, features, label, params, 'svm_photoz')
    print df_test['svm_photoz'].describe()
    ml_an.evaluate_regression(df_test['Z'],df_test['svm_photoz'])
    pz_an.plot_redshifts(df_test['Z'],df_test['svm_photoz'])
    pz_an.plot_error_hist(df_test['Z'],df_test['svm_photoz'])
    plt.show()



test_example()
# grid_search_example()
# predict_example()
