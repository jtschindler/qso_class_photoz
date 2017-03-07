import numpy as np
import pandas as pd
import math

from class_photoz import ml_quasar_sample as qs
from class_photoz import ml_sets as sets
from class_photoz import rf_class as rf_class
from class_photoz import rf_reg as rf_reg
from class_photoz import ml_analysis as ml_an
from class_photoz import photofit_analysis as pf_an


# 1) Photoz RF
# 2) Classification RF

# 3) Photofit full routine

def rf_full_emp(df_pred):
    # --------------------------------------------------------------------------
    # PHOTOMETRIC REDSHIFT ESTIMATION
    # --------------------------------------------------------------------------

    # Preparing the feature matrix

    df_train = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')

    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    df_train.replace(np.inf, np.nan,inplace=True)
    df_train = df_train.query('0 < Z_VI < 10')
    df_train.query('SDSS_mag_i <= 18.5',inplace=True)

    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)


    # Random Forest Regression Grid Search
    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    rand_state = 1
    params = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 4, 'n_jobs': 2, 'random_state':rand_state}

    df_pred = rf_reg.rf_reg_predict(df_train, df_pred, features, label, params, 'rf_emp_photoz')


    # --------------------------------------------------------------------------
    # QSO-STAR-CLASSIFICATION
    # --------------------------------------------------------------------------

    # Loading and preparing the data files

    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    df_quasars = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        # 'TMASS_j', \
                        # 'TMASS_h', \
                        # 'TMASS_k', \
                        'WISE_w1', \
                        'WISE_w2', \
                        ]

    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df_stars.query('SDSS_mag_i <= 18.5',inplace=True)
    df_quasars.query('SDSS_mag_i <= 18.5',inplace=True)


    print "Stars: ",df_stars.shape
    print "Quasars: ",df_quasars.shape


    # Preparing test and training sets

    #Create detailed classes
    df_quasars = qs.create_qso_labels(df_quasars, 'mult_class_true', 'z')
    df_stars = qs.create_star_labels(df_stars, 'mult_class_true', 'star_class')

    # Create binary classes
    df_quasars['bin_class_true']='QSO'
    df_stars['bin_class_true']='STAR'

    # Concatenate training set
    df_train = pd.concat([df_star,df_quasars])

    # Running the Random Forest method
    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz',  \
                'zw1', 'w1w2']


    label = 'mult_class_true'

    params = {'n_estimators': 300, 'max_depth': 25, 'min_samples_split': 4,
        'n_jobs': 2, 'random_state': 1}

    rand_state = 1

    clf,y_pred = rf_class_predict(df_train, df_pred, features, label, params,
                                                                rand_state)

    df_pred['rf_emp_mult_label_pred'] = y_pred

    df_pred['rf_emp_bin_class_pred'] = 'STAR'
    qso_query = 'rf_emp_mult_class_pred == "vlowz" or rf_emp_mult_class_pred == "lowz" or rf_emp_mult_class_pred == "midz" or rf_emp_mult_class_pred == "highz"'
    df_pred.loc[df_pred.query(qso_query).index,'rf_emp_bin_class_pred'] = 'QSO'

    return df_pred






def rf_full_sim(df_pred):
    # --------------------------------------------------------------------------
    # PHOTOMETRIC REDSHIFT ESTIMATION
    # --------------------------------------------------------------------------

    # Preparing the feature matrix

    df_train = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')

    passband_names = [\
            'SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
            # 'TMASS_j','TMASS_h','TMASS_k', \
            'WISE_w1','WISE_w2', \
            # 'WISE_w3' \
            ]

    # embed this in the sim qso conversion file!
    for name in passband_names:
        df_train.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_train.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)


    df_train.replace(np.inf, np.nan,inplace=True)

    df_train.query('obsMag_SDSS_i <= 18.5',inplace=True)

    df_train,features = qs.prepare_flux_ratio_catalog(df_train,passband_names)


    # Random Forest Regression Grid Search
    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']
    rand_state = 1
    params = {'n_estimators': 300, 'max_depth': 20, 'min_samples_split': 4, 'n_jobs': 2, 'random_state':rand_state}

    df_pred = rf_reg.rf_reg_predict(df_train, df_pred, features, label, params, 'rf_sim_photoz')


    # --------------------------------------------------------------------------
    # QSO-STAR-CLASSIFICATION
    # --------------------------------------------------------------------------

    # Loading and preparing the data files

    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    df_quasars = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        # 'TMASS_j', \
                        # 'TMASS_h', \
                        # 'TMASS_k', \
                        'WISE_w1', \
                        'WISE_w2', \
                        ]

    #embed this in the sim qso conversion file!
    for name in passband_names:
        df_quasars.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_quasars.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)


    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df_stars.query('SDSS_mag_i <= 18.5',inplace=True)
    df_quasars.query('obsMag_SDSS_i <= 18.5',inplace=True)


    print "Stars: ",df_stars.shape
    print "Quasars: ",df_quasars.shape


    # Preparing test and training sets

    #Create detailed classes
    df_quasars = qs.create_qso_labels(df_quasars, 'mult_class_true', 'z')
    df_stars = qs.create_star_labels(df_stars, 'mult_class_true', 'star_class')

    # Create binary classes
    df_quasars['bin_class_true']='QSO'
    df_stars['bin_class_true']='STAR'

    # Concatenate training set
    df_train = pd.concat([df_star,df_quasars])

    # Running the Random Forest method
    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz',  \
                'zw1', 'w1w2']


    label = 'mult_class_true'

    params = {'n_estimators': 300, 'max_depth': 25, 'min_samples_split': 4,
        'n_jobs': 2, 'random_state': 1}

    rand_state = 1

    clf,y_pred = rf_class_predict(df_train, df_pred, features, label, params,
                                                                rand_state)

    df_pred['rf_sim_mult_label_pred'] = y_pred

    df_pred['rf_sim_bin_class_pred'] = 'STAR'
    qso_query = 'rf_sim_mult_class_pred == "vlowz" or rf_sim_mult_class_pred == "lowz" or rf_sim_mult_class_pred == "midz" or rf_sim_mult_class_pred == "highz"'
    df_pred.loc[df_pred.query(qso_query).index,'rf_sim_bin_class_pred'] = 'QSO'

    return df_pred




def photofit_full_emp(df_pred):
    # Load the catalog from wich to make the star model
    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    df_stars.drop(df_stars.query('star_class == "null"').index, inplace=True)

    # Load the catalog from wich to make the quasar model
    df_qsos = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    df_qsos = df_qsos.query('0 <= Z_VI <= 10')
    print df_qsos.shape

    z_label = 'Z_VI'
    star_label = 'class_label'
    rand_state = 1

    params = {'binning' : 'minimum',
        'bin_param' : 50,
        'model_type' : 'median'}

    df_qsos = df_qsos.query('SDSS_mag_i < 18.5')
    df_stars = df_stars.query('SDSS_mag_i < 18.5')

    df_stars = qs.create_star_labels(df_stars, star_label, 'star_class')

    # Set binary and multi class columns for evaluation routines
    df_stars['bin_class_true'] = 'STAR'
    df_stars['mult_class_true'] = df_stars[star_label]
    df_qsos['bin_class_true'] = 'QSO'
    df_qsos = pf_an.set_redshift_classes(df_qsos, 'Z_VI', 'mult_class_true')

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            'WISE_w1',\
            'WISE_w2',\
            ]


    df_stars, features  =  qs.prepare_flux_ratio_catalog(df_stars, \
    passband_names, sigma=True)

    df_qsos, features  =  qs.prepare_flux_ratio_catalog(df_qsos, \
    passband_names, sigma=False)
    print df_qsos.shape, features


    df_train = pd.concat([df_stars,df_qsos])

    df_pred, qso_prob, qso_chisq = \
            photoz_fit(df_qsos,df_pred,features, z_label, params)

    df_pred, star_prob, star_chisq, star_model = \
            star_fit(df_stars, df_pred, features, star_label, params)

    # Classify the test set according to the lowest chi-squared value
    df_pred = pf_an.set_redshift_classes(df_pred, 'pf_photoz', 'pf_qso_class')

    df_pred = pf_an.set_pred_classes(df_pred)


    return df_pred
