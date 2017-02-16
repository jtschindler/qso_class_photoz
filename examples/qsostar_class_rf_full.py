import numpy as np
import pandas as pd
import math

from class_photoz import ml_quasar_sample as qs
from class_photoz import ml_sets as sets
from class_photoz import rf_class as rf_class


def create_labels(df_stars, df_quasars,z_label):

    df_stars['class_label'] = df_stars.star_class

    star_labels = df_stars.class_label.value_counts().index

    for label in star_labels:

        if df_stars.class_label.value_counts()[label] < 10:
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





def dr7dr12q_grid_search():

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



dr7dr12q_grid_search()
