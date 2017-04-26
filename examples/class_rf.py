import numpy as np
import pandas as pd
import math

import class_photoz

from class_photoz import ml_quasar_sample as qs
from class_photoz import ml_sets as sets
from class_photoz import rf_class as rf_class


def create_labels(df_stars, df_quasars,z_label):

    df_stars['class_label'] = df_stars.star_class

    star_labels = df_stars.class_label.value_counts().index

    for label in star_labels:

        if df_stars.class_label.value_counts()[label] < 400:
            df_stars.drop(df_stars.query('class_label == "'+label+'"').index,
                                                                inplace=True)

    lowz=[0,1.5,2.2,3.5]
    highz=[1.5,2.2,3.5,10]
    labels=['0<z<=1.5','1.5<z<=2.2','2.2<=3.5','3.5<z']
    df_quasars['class_label'] = 'null'
    df_quasars.query('0<'+str(z_label)+'<10',inplace=True)
    for idx in range(len(lowz)):

        df_quasars.loc[
                df_quasars.query(str(lowz[idx])+'<'+z_label+'<='+str(highz[idx])).index, \
                'class_label'] = labels[idx]

    print df_quasars.class_label.value_counts()
    print df_stars.class_label.value_counts()

    return df_stars,df_quasars



def grid_search_example():

    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    # df_quasars = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    df_quasars = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        'TMASS_j', \
                        'TMASS_h', \
                        'TMASS_k', \
                        'WISE_w1','WISE_w2', \
                        # 'WISE_w3', \
                        # 'WISE_w4', \
                        ]


    #TODO Only for now delete later
    df_stars = df_stars.rename(columns={'sigma_TMASS_ks':'sigma_TMASS_k', \
            'TMASS_ks':'TMASS_k','TMASS_mag_ks':'TMASS_mag_k'})


    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    #Reduce the total set of objects for testing the routines
    df_stars = df_stars.sample(frac=0.2)
    df_quasars = df_quasars.sample(frac=0.2)



    # Build a test sample with a given QSO to STAR ratio
    df = qs.build_full_sample(df_stars, df_quasars, 100)
    print df.label.value_counts()

    # Declare labels and select features to classify on
    labels = ["STAR","QSO"]
    features = ['WISE_w1','TMASS_j','jh', 'hk', 'kw1', 'w1w2']
    # features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh',  \
    #             'hk', 'kw1', 'w1w2']
    # features = ['SDSS_i','TMASS_j','ug','gr','ri','iz','zj','jh',  \
    #                 'hk']
    # features = ['SDSS_i','ug','gr','ri','iz']
    label = 'label'

    param_grid = [{'n_estimators': [25,50,100,200], 'min_samples_split': [2,3,4],
                    'max_depth' : [10,15,20]}]

    rf_class.rf_class_grid_search(df,features, label ,param_grid)


def test_example():

    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    df_quasars = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    #df_quasars = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        'TMASS_j', \
                        'TMASS_h', \
                        'TMASS_k', \
                        'WISE_w1', \
                        'WISE_w2', \
                        # 'WISE_w3', \
                        # 'WISE_w4', \
                        ]


    #TODO Only for now delete later
    #df_stars = df_stars.rename(columns={'sigma_TMASS_ks':'sigma_TMASS_k', \
    #        'TMASS_ks':'TMASS_k','TMASS_mag_ks':'TMASS_mag_k'})

    #embed this in the sim qso conversion file!
    # for name in passband_names:
    #     df_quasars.rename(columns={'obsFlux_'+name:name},inplace=True)
    #     df_quasars.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)


    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)


    #Reduce the total set of objects for testing the routines
    df_stars = df_stars.sample(frac=0.2)
    df_quasars = df_quasars.sample(frac=0.2)


    df_stars.query('SDSS_mag_i <= 18.5',inplace=True)
    df_quasars.query('SDSS_mag_i <=18.5',inplace=True)
    #df_quasars.query('obsMag_SDSS_i <= 18.5',inplace=True)
    print "Stars: ",df_stars.shape
    print "Quasars: ",df_quasars.shape

    #Create more detailed classes
    df_stars, df_quasars = create_labels(df_stars, df_quasars,'z')


    #Create detailed classes
    df_quasars = qs.create_qso_labels(df_quasars, 'mult_class_true', 'z')
    df_stars = qs.create_star_labels(df_stars, 'mult_class_true', 'star_class')

    # Create binary classes
    df_quasars['bin_class_true']='QSO'
    df_stars['bin_class_true']='STAR'

    # Make test and training set
    df_train, df_pred = qs.make_train_pred_set(df_stars, df_quasars, 0.2, rand_state = 1)


    features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh',  \
                'hk', 'kw1', 'w1w2']
    # features = ['SDSS_i','TMASS_j','ug','gr','ri','iz','zj','jh', 'hk']
    #features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    #features = ['SDSS_i','ug','gr','ri','iz']

    label = 'mult_class_true'


    params = {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 3,
        'n_jobs': 4, 'random_state': 1}

    rand_state=1

    # y_true, y_pred, df_prob = rf_class.rf_class_example(df_train, df_pred, features, label, params,rand_state)
    y_true, y_pred, y_prob = rf_class.rf_class_predict(df_train, df_pred, features, label, params,rand_state)

    print y_prob
    # df_prob.to_hdf('df_prob.hdf5','data')





test_example()
