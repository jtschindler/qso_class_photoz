import numpy as np
import pandas as pd
import math

from sklearn.metrics import confusion_matrix

from class_photoz import ml_quasar_sample as qs
from class_photoz import ml_sets as sets
from class_photoz import rf_class as rf_class
from class_photoz import ml_analysis as ml_an
from class_photoz import photofit_analysis as pf_an


def dr7dr12q_grid_search():

    # --------------------------------------------------------------------------
    # Read data file and input parameters
    # --------------------------------------------------------------------------

    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    df_quasars = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    df_stars.dropna(subset=['star_class'],inplace=True)

    param_grid = [{'n_estimators': [100,200,300], 'min_samples_split': [2,3,4],
                    'max_depth' : [15,20,25]}]
    # param_grid = [{'n_estimators': [100], 'min_samples_split': [2],
                    # 'max_depth' : [20]}]
    rand_state=1
    scores = ['f1_weighted']

    # Restrict the data set
    df_stars.query('SDSS_mag_i <= 18.5',inplace=True)
    df_quasars.query('SDSS_mag_i <= 18.5',inplace=True)

    # Create basic classes
    df_quasars['label']='QSO'
    df_stars['label']='STAR'

    #Create more detailed classes
    df_quasars = qs.create_qso_labels(df_quasars, 'class_label', 'Z_VI')
    df_stars = qs.create_star_labels(df_stars, 'class_label', 'star_class')


    # FOR TESTING PURPOSES
    # df_stars = df_stars.sample(frac=0.2)
    # df_quasars = df_quasars.sample(frac=0.2)

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        # 'TMASS_j', \
                        # 'TMASS_h', \
                        # 'TMASS_k', \
                        # 'WISE_w1','WISE_w2', \
                        ]

    df_stars_train = df_stars.copy(deep=True)
    df_qsos_train = df_quasars.copy(deep=True)

    label = 'class_label'

    df_stars_train,features = qs.prepare_flux_ratio_catalog(df_stars_train,passband_names)
    df_qsos_train,features = qs.prepare_flux_ratio_catalog(df_qsos_train,passband_names)

    df_train, df_pred = qs.make_train_pred_set(df_stars_train, df_qsos_train, 0.2 ,rand_state, 'i19_5_')

    #Choose label: 'label' = 2 classes, 'class_label'= multiple classes


    features = ['SDSS_i','ug','gr','ri','iz']

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    rf_class.rf_class_grid_search(df_train, df_pred, features, label ,param_grid, rand_state, scores, 'test')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------



    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        # 'TMASS_j', \
                        # 'TMASS_h', \
                        # 'TMASS_k', \
                        'WISE_w1','WISE_w2', \
                        ]

    label = 'class_label'


    df_stars_train = df_stars.copy(deep=True)
    df_qsos_train = df_quasars.copy(deep=True)

    df_stars_train,features = qs.prepare_flux_ratio_catalog(df_stars_train,passband_names)
    df_qsos_train,features = qs.prepare_flux_ratio_catalog(df_qsos_train,passband_names)

    df_train, df_pred = qs.make_train_pred_set(df_stars_train, df_qsos_train, 0.2 ,rand_state)
    #Choose label: 'label' = 2 classes, 'class_label'= multiple classes

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    rf_class.rf_class_grid_search(df_train, df_pred, features, label ,param_grid, rand_state, scores, 'test')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------



    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        'TMASS_j', \
                        'TMASS_h', \
                        'TMASS_k', \
                        'WISE_w1','WISE_w2', \
                        ]

    label = 'class_label'


    df_stars_train = df_stars.copy(deep=True)
    df_qsos_train = df_quasars.copy(deep=True)

    df_stars_train,features = qs.prepare_flux_ratio_catalog(df_stars_train,passband_names)
    df_qsos_train,features = qs.prepare_flux_ratio_catalog(df_qsos_train,passband_names)

    df_train, df_pred = qs.make_train_pred_set(df_stars_train, df_qsos_train, 0.2 ,rand_state)

    #Choose label: 'label' = 2 classes, 'class_label'= multiple classes


    features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh',  \
                'hk', 'kw1', 'w1w2']

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    rf_class.rf_class_grid_search(df_train, df_pred, features, label ,param_grid, rand_state, scores, 'SDSSTMASSW1W2_i195')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------





def simqsos_grid_search():


    # TODO This needs to be adjusted for the simulated QSOS


    # --------------------------------------------------------------------------
    # Read data file and input parameters
    # --------------------------------------------------------------------------

    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    df_quasars = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')

    param_grid = [{'n_estimators': [50,100,200,300], 'min_samples_split': [2,3,4],
                    'max_depth' : [15,20,25]}]
    rand_state=1
    scores = ['f1_weighted']

    # Restrict the data set
    df_stars.query('SDSS_mag_i <= 18.5',inplace=True)
    df_quasars.query('SDSS_mag_i <=18.5',inplace=True)

    # Create basic classes
    df_quasars['label']='QSO'
    df_stars['label']='STAR'

    #Create more detailed classes
    df_stars, df_quasars = create_labels(df_stars, df_quasars,'Z_VI')

    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        # 'TMASS_j', \
                        # 'TMASS_h', \
                        # 'TMASS_k', \
                        # 'WISE_w1','WISE_w2', \
                        ]

    df_stars_train = df_stars.copy(deep=True)
    df_qsos_train = df_quasars.copy(deep=True)

    df_stars_train,features = qs.prepare_flux_ratio_catalog(df_stars_train,passband_names)
    df_qsos_train,features = qs.prepare_flux_ratio_catalog(df_qsos_train,passband_names)

    df = pd.concat([df_stars_train,df_qsos_train])

    #Choose label: 'label' = 2 classes, 'class_label'= multiple classes
    label = 'class_label'

    features = ['SDSS_i','ug','gr','ri','iz']

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    rf_class.rf_class_grid_search(df,features, label ,param_grid, rand_state, scores, 'SDSS')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------



    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        # 'TMASS_j', \
                        # 'TMASS_h', \
                        # 'TMASS_k', \
                        'WISE_w1','WISE_w2', \
                        ]

    df_stars_train = df_stars.copy(deep=True)
    df_qsos_train = df_quasars.copy(deep=True)

    df_stars_train,features = qs.prepare_flux_ratio_catalog(df_stars_train,passband_names)
    df_qsos_train,features = qs.prepare_flux_ratio_catalog(df_qsos_train,passband_names)

    df = pd.concat([df_stars_train,df_qsos_train])

    #Choose label: 'label' = 2 classes, 'class_label'= multiple classes
    label = 'class_label'

    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz','zw1','w1w2']

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    rf_class.rf_class_grid_search(df,features, label ,param_grid, rand_state, scores, 'SDSSW1W2')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------



    # --------------------------------------------------------------------------
    # Preparation of training set
    # --------------------------------------------------------------------------

    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        'TMASS_j', \
                        'TMASS_h', \
                        'TMASS_k', \
                        'WISE_w1','WISE_w2', \
                        ]

    df_stars_train = df_stars.copy(deep=True)
    df_qsos_train = df_quasars.copy(deep=True)

    df_stars_train,features = qs.prepare_flux_ratio_catalog(df_stars_train,passband_names)
    df_qsos_train,features = qs.prepare_flux_ratio_catalog(df_qsos_train,passband_names)

    df = pd.concat([df_stars_train,df_qsos_train])

    #Choose label: 'label' = 2 classes, 'class_label'= multiple classes
    label = 'class_label'

    features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh',  \
                'hk', 'kw1', 'w1w2']

    # --------------------------------------------------------------------------
    # Random Forest Regression Grid Search
    # --------------------------------------------------------------------------

    rf_class.rf_class_grid_search(df,features, label ,param_grid, rand_state, scores, 'SDSSTMASSW1W2')

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------





def full_test():
    # --------------------------------------------------------------------------
    # Loading and preparing the data files
    # --------------------------------------------------------------------------
    df_stars = pd.read_hdf('../class_photoz/data/DR13_stars_clean_flux_cat.hdf5','data')
    df_quasars = pd.read_hdf('../class_photoz/data/DR7DR12Q_clean_flux_cat.hdf5','data')
    # df_quasars = pd.read_hdf('../class_photoz/data/brightqsos_sim_2k_new.hdf5','data')
    # df_stars.drop(df_stars.query('star_class == "null"').index, inplace=True)
    passband_names = ['SDSS_u','SDSS_g','SDSS_r','SDSS_i','SDSS_z', \
                        # 'TMASS_j', \
                        # 'TMASS_h', \
                        # 'TMASS_k', \
                        'WISE_w1', \
                        'WISE_w2', \
                        ]

    # #embed this in the sim qso conversion file!
    for name in passband_names:
        df_quasars.rename(columns={'obsFlux_'+name:name},inplace=True)
        df_quasars.rename(columns={'obsFluxErr_'+name:'sigma_'+name},inplace=True)

    df_stars,features = qs.prepare_flux_ratio_catalog(df_stars,passband_names)
    df_quasars,features = qs.prepare_flux_ratio_catalog(df_quasars,passband_names)

    df_stars.query('SDSS_mag_i <= 18.5',inplace=True)
    df_quasars.query('SDSS_mag_i <= 18.5',inplace=True)
    # df_quasars.query('obsMag_SDSS_i <= 18.5',inplace=True)

    print "Stars: ",df_stars.shape
    print "Quasars: ",df_quasars.shape

    # --------------------------------------------------------------------------
    # Preparing test and training sets
    # --------------------------------------------------------------------------
    #Create detailed classes
    df_quasars = qs.create_qso_labels(df_quasars, 'mult_class_true', 'z')
    df_stars = qs.create_star_labels(df_stars, 'mult_class_true', 'star_class')

    # Create binary classes
    df_quasars['bin_class_true']='QSO'
    df_stars['bin_class_true']='STAR'

    # Make test and training set
    df_train, df_pred = qs.make_train_pred_set(df_stars, df_quasars, 0.2, rand_state = 1)
    print df_train.shape, df_pred.shape
    # --------------------------------------------------------------------------
    # Running the Random Forest method
    # --------------------------------------------------------------------------

    # features = ['SDSS_i','WISE_w1','TMASS_j','ug','gr','ri','iz','zj','jh',  \
    #             'hk', 'kw1', 'w1w2']
    features = ['SDSS_i','WISE_w1','ug','gr','ri','iz',  \
                'zw1', 'w1w2']
    # features = ['SDSS_i','ug','gr','ri','iz']

    label = 'mult_class_true'

    params = {'n_estimators': 300, 'max_depth': 25, 'min_samples_split': 4,
        'n_jobs': 2, 'random_state': 1}
    print features
    print params
    rand_state=1

    y_true, y_pred = \
        rf_class.rf_class_example(df_train, df_pred, features, label, params,rand_state)

    # --------------------------------------------------------------------------
    # Additional analysis
    # --------------------------------------------------------------------------

    data = {'mult_class_true':y_true,'mult_class_pred':y_pred, 'bin_class_true':df_pred.bin_class_true}
    df = pd.DataFrame(data)

    df['bin_class_pred'] = 'STAR'
    qso_query = 'mult_class_pred == "vlowz" or mult_class_pred == "lowz" or mult_class_pred == "midz" or mult_class_pred == "highz"'
    df.loc[df.query(qso_query).index,'bin_class_pred'] = 'QSO'

    labels = ('QSO','STAR')
    y_true = df.bin_class_true.values
    y_pred = df.bin_class_pred.values

    pf_an.classification_analysis(y_true,y_pred,labels)

dr7dr12q_grid_search()
# load_file_grid_search()
# full_test()
