import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline


def prepare_flux_ratio_catalog(df,passband_names,sigma=False):
    """ Calculating the flux ratios from the fluxes provided by
        the input df and dropping all rows with NaN values in the
        process to ensure a full data set.

        Parameters:
            df : dataframe
            Dataframe containing the features

            passband_names : list of strings
            Names of the passbands (features) to be considered

        Returns:
            model_catalog : dataframe
            Returns the model catalog without NaN values in all features

            flux_ratio_names : list of strings
            Names of the flux ratios calculated
        """

    # Drop all rows with NaN values in the passband considered

    for name in passband_names:
        df.dropna(axis=0,how='any',subset=['sigma_'+name],inplace=True)
        print df.shape

    df.dropna(axis=0,how='any',subset=passband_names,inplace=True)
    print df.shape

    # Calculate the flux ratios and add them to the dataframe
    flux_ratio_names = []
    flux_ratio_err_names= []

    if sigma :

        for name in passband_names:
            df.dropna(axis=0,how='any',subset=['sigma_'+name],inplace=True)

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














# def build_spline_model(model_catalog, flux_ratio_names):
#
#   #drop ridicoulus values
#   for i in flux_ratio_names:
#     to_drop = model_catalog.query(i+'<-100 or '+i+'>100').index
#     model_catalog.drop(to_drop,inplace=True)
#
#
#   objects_per_bin = 400
#
#   # copy redshift columns from dataframe DataFrame
#   redshifts = model_catalog[['redshift']]
#
#   redshifts = redshifts.sort_values('redshift')
#
#   #choose redshift range for minimum binning
#   redshifts = redshifts.query('0.2 <= redshift <= 4.0')
#
#   # generate bin borders
#   a = np.array_split(np.array(redshifts),redshifts.shape[0]/objects_per_bin)
#
#
#   bins =np.arange(0,0.2,0.05)
#
#   # calculate lower bin border
#   for i in range(len(a)-1):
#       bins = np.append(bins,(a[i].max() + a[i+1].min())/2.)
#
#   bins = np.append(bins,a[len(a)-1].max())
#
#   bins = np.append(bins,np.arange(4,6.0,0.1))
#
#   #bins = np.array(bins)
#
#   # calculating the middle redshift and bin width
#   bin_middle = (bins[:-1]+bins[1:])/2.
#   bin_width = bins[1:]-bins[:-1]
#
#   # create array that holds all information about the bins
#   # lower edge, upper edge, middle redshift, redshift width
#   bin_data = np.array([bins[:-1],bins[1:],bin_middle,bin_width])
#
#   # Generate pandas groups from which we can calculate the colors
#   redshift_bins = model_catalog.groupby(pd.cut(model_catalog.redshift, bins))
#
#   flux_model = build_binned_model(redshift_bins,bin_data,flux_ratio_names)
#
#   spline_list = []
  #
  #
  # for i in range(len(flux_ratio_names)):
  #
  #     spl = UnivariateSpline(np.array(flux_model.bin_middle), np.array(flux_model[flux_ratio_names[i]]))
  #
  #
  #     spline_list.append(spl)
  #
  #
  #
  # return model_catalog,spline_list,flux_model


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
