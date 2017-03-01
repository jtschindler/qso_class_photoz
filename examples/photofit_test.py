import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from class_photoz import ml_quasar_sample as qs
from class_photoz.photofit_photoz import *
from class_photoz.photofit import *
from class_photoz import photofit_analysis as pf_an
from class_photoz.photofit_find_star_class import *
from sklearn.model_selection import train_test_split

def full_analysis(df_test):
    # df_test = pd.read_hdf('test_set.hdf5','data')
    # Photoz

    df_test = pf_an.set_pred_classes(df_test)

    print df_test.mult_class_true.value_counts()
    print df_test.mult_class_pred.value_counts()

    print df_test.bin_class_true.value_counts()
    print df_test.bin_class_pred.value_counts()

    pf_an.photoz_analysis(df_test, 'pf_photoz', 'Z_VI')

    pf_an.photoz_analysis(df_test, 'peak_a_mode', 'Z_VI')

    print df_test.mult_class_true.value_counts()
    print df_test.mult_class_pred.value_counts()

    print df_test.bin_class_true.value_counts()
    print df_test.bin_class_pred.value_counts()

    #  Binary Class
    y_true = df_test.bin_class_true.values.astype('string')
    y_pred = df_test.bin_class_pred.values.astype('string')
    labels = ('QSO','STAR')
    pf_an.classification_analysis(y_true,y_pred,labels)

    #  Multiple Classes
    y_true = df_test.mult_class_true.values.astype('string')
    y_pred = df_test.mult_class_pred.values.astype('string')
    labels = df_test.mult_class_true.value_counts().index

    pf_an.classification_analysis(y_true,y_pred,labels)


    plt.show()

def test_photoz():

    # Load the catalog from wich to make the quasar model
    df = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')

    df = df.query('0 < Z_VI < 10')
    df = df.query('SDSS_mag_i < 17.5')
    # df = df.sample(frac=0.2)

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            'WISE_w1',\
            'WISE_w2']


    df, features  =  qs.prepare_flux_ratio_catalog(df, \
    passband_names, sigma=True)


    label = 'Z_VI'

    params = {'binning' : 'minimum',
        'bin_param' : 100,
        'model_type' : 'median'}

    photoz_fit_test(df,features,label,params,rand_state = 1, save_data=True,
     save_name = 'test2')





def test_star_fit():

    # Load the catalog from wich to make the quasar model
    df = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')

    df.drop(df.query('star_class == "null"').index, inplace=True)

    df = df.query('SDSS_mag_i < 17.5')

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            'WISE_w1',\
            'WISE_w2']

    df, features  =  qs.prepare_flux_ratio_catalog(df, \
    passband_names, sigma=True)

    params = {'binning' : 'minimum',
        'bin_param' : 100,
        'model_type' : 'median'}

    label = 'star_class'

    star_fit_test(df, features, label, params, rand_state = 1, save_data=True,
     save_name = 'test')




def test_full_fit():

    # Load the catalog from wich to make the star model
    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    df_stars.drop(df_stars.query('star_class == "null"').index, inplace=True)

    # Load the catalog from wich to make the quasar model
    df_qsos = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    df_qsos = df_qsos.query('0 < Z_VI < 10')

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
    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        # 'TMASS_j', \
                        # 'TMASS_h', \
                        # 'TMASS_k', \
                        # 'WISE_w1','WISE_w2', \
                        ]

    df_stars, features  =  qs.prepare_flux_ratio_catalog(df_stars, \
    passband_names, sigma=True)
    df_qsos, features  =  qs.prepare_flux_ratio_catalog(df_qsos, \
    passband_names, sigma=True)

    df_train_stars, df_train_qsos, df_test = \
            qs.make_train_pred_set(df_stars, df_qsos, 0.2, rand_state, 'SDSSTMASSW1W2_emp_i18_5_',
                                                    concat=False, save = True)

    print df_train_stars.mult_class_true.value_counts()
    print df_train_qsos.mult_class_true.value_counts()
    print df_test.mult_class_true.value_counts()



    df_test, qso_prob, qso_chisq = \
            photoz_fit(df_train_qsos,df_test,features, z_label, params)

    df_test, star_prob, star_chisq, star_model = \
            star_fit(df_train_stars, df_test, features, star_label, params)

    # Classify the test set according to the lowest chi-squared value
    df_test = pf_an.set_redshift_classes(df_test, 'pf_photoz', 'qso_class')

    df_test = pf_an.set_pred_classes(df_test)

    df_test.to_hdf('photofit_SDSS_emp_i18_5.hdf5','data')

    full_analysis(df_test)




df = pd.read_hdf('photofit_SDSSTMASSW1W2_emp_i18_5.hdf5','data')
full_analysis(df)
# test_full_fit()

# test_star_fit()
# test_photoz()
# bin_middle, bin_width, pdf, bins = make_mock_prob_pdf()
# calc_prob_peaks(bin_middle, bin_width, pdf, bins )
