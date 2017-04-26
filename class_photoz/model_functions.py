import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline




def build_full_sample(df_stars, df_quasars, star_qso_ratio,return_cats=False):

    """ Merging the star and quasar flux_ratio catalogs according to
    the set variable star_quasar_ratio. This is the first step to create
    more realistic data set, since the intrinsic ratio of stars to quasars
    will not be mimicked by simply combining both data sets. The catalogs
    are labelled dependend on their origin catalog.

    TO DO:
    This function should be expanded in order to return a DataFrame that
    mimicks the intrinsic quasar/star distribution as good as possible.

    Parameters:
            df_stars : pandas dataframe
            Star flux ratio catalog

            df_quasars : pandas dataframe
            Quasar flux ratio catalog

            star_qso_ratio : integer
            Goal ratio of stars to quasars

    Returns:
            df : pandas dataframe
            Merged flux ratio catalog with specified star to quasar ratio
    """

    df_quasars['label'] = 'QSO'
    df_stars['label'] = 'STAR'

    if return_cats:
        if df_stars.shape[0] > df_quasars.shape[0]*star_qso_ratio:
            # calculate number of objects to sample
            sample_size = df_quasars.shape[0]
            star_sample = df_stars.sample(sample_size*star_qso_ratio)
            qso_sample = df_quasars.sample(sample_size)

            df = pd.concat([qso_sample,star_sample])
        else:
            # calculate number of objects to sample
            sample_size = df_stars.shape[0]
            star_sample = df_stars.sample(sample_size)
            qso_sample = df_quasars.sample(sample_size/star_qso_ratio)
            df = pd.concat([qso_sample,star_sample])


        star_cat = df_stars.drop(star_sample.index)
        qso_cat = df_quasars.drop(qso_sample.index)

        return df, star_cat, qso_cat

    else :
        if df_stars.shape[0] > df_quasars.shape[0]*star_qso_ratio:
            # calculate number of objects to sample
            sample_size = df_quasars.shape[0]
            star_sample = df_stars.sample(sample_size*star_qso_ratio)
            qso_sample = df_quasars.sample(sample_size)

            df = pd.concat([qso_sample,star_sample])
        else:
            # calculate number of objects to sample
            sample_size = df_stars.shape[0]
            star_sample = df_stars.sample(sample_size)
            qso_sample = df_quasars.sample(sample_size/star_qso_ratio)
            df = pd.concat([qso_sample,star_sample])

        return df



def build_star_model(star_catalog, features):
    """ This function prepares the stellar model catalog by usign the
    information about the stellar classification to build bins and then
    calculates the median value of all the features for each bin.
    THE STAR CATALOG NEEDS TO INCLUDE A COLUMN CALLED star_class FOR THE STELLAR
    CLASSIFICATION.

        Parameters:
            star_catalog : dataframe
            Dataframe containing all the feature information as well as the
            stellar classification to build the stellar feature_model.

            features : list of string
            Names of the features considered (flux_ratio_names, color_names)

        Returns:
            star_model : Dataframe containing the median values of all the
            features as well as the information about the bins
        """


    # Generate pandas groups from which we can calculate the colors
    star_bins = star_catalog.groupby(star_catalog.star_class)

    # copy star class columns from star catalog to model DataFrame
    star_model = pd.DataFrame(star_bins.agg([np.median])['ug'].index,columns=[ \
                                                                'star_class'])

    # add number of objects per star class as a column
    star_model['class_counts'] = star_bins.count()['class_sdss'].values

    #star_classes = [WD,\
                  #O,O8,O9,OB,B0,B1,B2,B3,B5,B6,B7,B8,B9,\
                  #A0,A1,A2,A3,A4,A5,A6,A8,A9,\
                  #F0,F2,F3,F5,F6,F8,F9,\
                  #G0,G1,G2,G3,G4,G5,G8,G9,\
                  #K0,K1,K2,K3,K4,K5,K7, \
                  #L0,L1,L2,L3,L4,L5,L9,Ldwarf, \
                  #M0,M1,M2,M3,M4,M5,M6,M7,M8,M9, \
                  #C, T]

    for i in range(len(features)):
      star_model[features[i]] = np.array( star_bins.agg([np.median])[features[i]])

    return star_model
