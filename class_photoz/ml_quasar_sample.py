import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, cross_validation

def prepare_flux_ratio_catalog(df,passband_names,sigma=False):
    """ Calculating the flux ratios from the fluxes provided by
        the input df and dropping all rows with NaN values in the
        process to ensure a full data set

    Input:
            df (DataFrame) as the input flux catalog
            passband_names (list) of the filter names considered
                for calculating the flux ratios
    Output:
            df (DataFrame) catalog including the flux ratios
            flux_ratio_names (list) list of the labels for
                the flux ratio columns
    """

    # Drop all rows with NaN values in the passband considered
    df.dropna(axis=0,how='any',subset=passband_names,inplace=True)
    

    # Calculate the flux ratios and add them to the dataframe
    flux_ratio_names = []
    flux_ratio_err_names= []

    for name in passband_names:
        df.dropna(axis=0,how='any',subset=['sigma_'+name],inplace=True)

    if sigma :

        for i in range(len(passband_names)-1):

            passband_a = np.array(df[passband_names[i]])
            passband_b = np.array(df[passband_names[i+1]])
            sigma_a = np.array(df['sigma_'+passband_names[i]])
            sigma_b = np.array(df['sigma_'+passband_names[i+1]])

            passband_a_name = passband_names[i].split('_')[1]
            passband_b_name = passband_names[i+1].split('_')[1]

            df[str(passband_a_name+passband_b_name)] = \
            passband_a / passband_b

            flux_ratio_names.append(str(passband_a_name+passband_b_name))

            df[str('sigma_'+passband_a_name+passband_b_name)] = \
            np.sqrt((sigma_a/passband_b)**2 + (passband_a/passband_b**2*sigma_b))

            flux_ratio_err_names.append('sigma_'+ \
            str(passband_a_name+passband_b_name))

    else :
        for i in range(len(passband_names)-1):

            passband_a = np.array(df[passband_names[i]])
            passband_b = np.array(df[passband_names[i+1]])
            # sigma_a = np.array(df['sigma_'+passband_names[i]])
            # sigma_b = np.array(df['sigma_'+passband_names[i+1]])

            passband_a_name = passband_names[i].split('_')[1]
            passband_b_name = passband_names[i+1].split('_')[1]

            df[str(passband_a_name+passband_b_name)] = \
            passband_a / passband_b

            flux_ratio_names.append(str(passband_a_name+passband_b_name))

            # df[str('sigma_'+passband_a_name+passband_b_name)] = \
            # np.sqrt((sigma_a/passband_b)**2 + (passband_a/passband_b**2*sigma_b))

            # flux_ratio_err_names.append('sigma_'+ \
            # str(passband_a_name+passband_b_name))




    return df, flux_ratio_names


def build_full_sample(df_stars, df_quasars, star_qso_ratio):

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
