import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from class_photoz import ml_quasar_sample as qs
from class_photoz.photofit_photoz import *
from class_photoz.photofit import *
from class_photoz import photoz_analysis as pz_an
from class_photoz import photofit_analysis as pf_an
from class_photoz.photofit_find_star_class import *
from sklearn.model_selection import train_test_split
import sklearn.metrics as met

# def create_star_qso_testset(df_stars, df_qsos, features, rand_state):
#
#     df_train_stars, df_stars_test = train_test_split(
#             df_stars, test_size=0.2,random_state=rand_state)
#
#     df_train_qsos, df_qsos_test = train_test_split(
#             df_qsos, test_size=0.2,random_state=rand_state)
#
#     df_test = pd.concat((df_qsos_test,df_stars_test),ignore_index = True)
#
#
#     return df_train_stars, df_train_qsos, df_test



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
    star_label = 'star_class'

    params = {'binning' : 'minimum',
        'bin_param' : 100,
        'model_type' : 'median'}

    df_qsos = df_qsos.query('SDSS_mag_i < 17.0')
    df_stars = df_stars.query('SDSS_mag_i < 17.0')

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            'WISE_w1',\
            'WISE_w2']

    # Set binary and multi class columns for evaluation routines
    df_stars['bin_class_true'] = 'STAR'
    df_stars['mult_class_true'] = df_stars['star_class']
    df_qsos['bin_class_true'] = 'QSO'
    df_qsos = pf_an.set_redshift_classes(df_qsos, 'Z_VI', 'mult_class_true')

    df_stars, features  =  qs.prepare_flux_ratio_catalog(df_stars, \
    passband_names, sigma=True)
    df_qsos, features  =  qs.prepare_flux_ratio_catalog(df_qsos, \
    passband_names, sigma=True)

    rand_state = 1

    df_train_stars, df_train_qsos, df_test = \
            qs.make_train_pred_set(df_stars, df_qsos, 0.2, rand_state,
                                                    concat=False, save = False)


    df_test, qso_prob, qso_chisq = \
            photoz_fit(df_train_qsos,df_test,features, z_label, params)

    df_test, star_prob, star_chisq, star_model = \
            star_fit(df_train_stars, df_test, features, star_label, params)

    # Classify the test set according to the lowest chi-squared value

    df_test = pf_an.set_redshift_classes(df_test, 'pf_photoz', 'qso_class')

    df_test.to_hdf('test_set.hdf5','data')

    print df_test.qso_class.value_counts()
    print df_test.mult_class_true.value_counts()

    df_test = pf_an.set_pred_classes(df_test)

    # df_test.loc[
    #     df_test.query('pf_qso_redchisq < pf_star_redchisq').index ,
    #      'mult_class_pred'] = \
    #      df_test.query('pf_qso_redchisq < pf_star_redchisq')['qso_class']
    #
    # df_test.loc[
    #     df_test.query('pf_qso_redchisq >= pf_star_redchisq').index ,
    #      'mult_class_pred'] = \
    #      df_test.query('pf_qso_redchisq >= pf_star_redchisq')['pf_star_class']
    #
    # df_test.loc[df_test.query('pf_qso_redchisq < pf_star_redchisq').index ,
    #  'bin_class_pred'] = 'QSO'
    #
    # df_test.loc[df_test.query('pf_qso_redchisq >= pf_star_redchisq').index ,
    #  'bin_class_pred'] = 'STAR'

    df_test.to_hdf('test_set.hdf5','data')

    # Analysis of the chi-squared fitting

    #  a) Photometric redshift analysis
    qso_test = df_test.query('bin_class_true == "QSO"')
    z_true = qso_test[z_label].values
    z_pred = qso_test.pf_photoz.values

    print r2_score(z_true, z_pred)

    pz_an.plot_redshifts(y_true,y_pred)
    pz_an.plot_error_hist(y_true,y_pred)

    #  b) Classification analysis multiple classes

    #  c) Classification analysis QSO/STAR


def full_analysis():
    df_test = pd.read_hdf('test_set.hdf5','data')
    # Photoz

    # z_label = 'Z_VI'
    # qso_test = df_test.query('bin_class_true == "QSO"')
    # print qso_test.shape
    # z_true = qso_test[z_label].values
    # z_pred = qso_test.pf_photoz.values
    #
    # print r2_score(z_true, z_pred)
    #
    # pz_an.plot_redshifts(z_true,z_pred)
    # pz_an.plot_error_hist(z_true,z_pred)
    #
    # plt.show()

    #  Bin Class
    y_true = df_test.bin_class_true.values.astype('string')
    y_pred = df_test.bin_class_pred.values.astype('string')


    print met.precision_recall_fscore_support(y_true,y_pred, labels=('QSO','STAR'))
    print met.precision_recall_fscore_support(y_true,y_pred, labels=('QSO','STAR'),average='weighted')




    cnf_matrix = confusion_matrix(y_true, y_pred, labels=['QSO','STAR'], sample_weight=None)

    ml_an.plot_confusion_matrix(cnf_matrix, classes=['QSO','STAR'], normalize=False,
                      title='Confusion matrix, with normalization')

    #  multi Class
    y_true = df_test.mult_class_true.values.astype('string')
    y_pred = df_test.mult_class_pred.values.astype('string')

    # class_names =  df_test.mult_class_true.value_counts().index

    class_names = ['WD','O','OB','B','A','F','G','K','M','L','T','CV','Carbon','vlowz','lowz','midz','highz']


    print met.precision_recall_fscore_support(y_true,y_pred, labels=class_names)
    print met.precision_recall_fscore_support(y_true,y_pred, labels=class_names, average='weighted')

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=class_names, sample_weight=None)

    ml_an.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')



    plt.show()



test_full_fit()
full_analysis()
# test_star_fit()
# test_photoz()
# bin_middle, bin_width, pdf, bins = make_mock_prob_pdf()
# calc_prob_peaks(bin_middle, bin_width, pdf, bins )
