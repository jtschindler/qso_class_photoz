import os
import pandas as pd
import numpy as np
from time import gmtime, strftime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score

from class_photoz import photofit_photoz as pfz
from class_photoz import photoz_analysis as pz_an
from class_photoz import photofit_find_star_class as pfsc

from class_photoz import ml_analysis as ml_an

def create_simple_bins(df, bin_size, z_label):
    """ This function prepares the model catalog to be binned by a very simple
    model with four different bin sizes (z<2 : bin_size, 2<z<3 : bin_size x2,
    3<z<5 : bin_size x4, 5<z<6 : bin_size x10).

        Parameters:
            df : dataframe
            Dataframe containing the features

            bin_size : float
            Base bin size

        Returns:
            redshift_bins : groupby element
            This groupby elements contains all values from the model catalog
            grouped by the redshift bins.


            bin_data : array-like ([ number of bins ]x4)
            This array contains the information about the bins. It contains the
            lower redshift boundary, the upper redshift boundary, the mean
            redshift and the redshift width in this order.

            bins : array-like
            The bins calculated by this method

        """

    # below z=2
    bin_size_low = bin_size #0.025

    # 2 < z < 3
    bin_size_med = bin_size*2

    # 3 < z < 5
    bin_size_high = bin_size*4

    # 5 < z < 6
    bin_size_highest = bin_size*10

    # building the bin list
    bins = np.arange(0,0.5,bin_size_high)
    bins = np.append(bins,np.arange(0.5,2,bin_size_low))
    bins = np.append(bins,np.arange(2,3,bin_size_med))
    bins = np.append(bins,np.arange(3,5,bin_size_high))
    bins = np.append(bins,np.arange(5,6,bin_size_highest))

    # calculating the middle redshift and bin width
    bin_middle = np.array((bins[:-1]+bins[1:])/2.)
    bin_width = np.array(bins[1:]-bins[:-1])

    # create array that holds all information about the bins
    # lower edge, upper edge, middle redshift, redshift width
    bin_data = np.array([bins[:-1],bins[1:],bin_middle,bin_width])

    # Generate pandas groups from which we can calculate the colors
    redshift_bins = df.groupby(pd.cut(df[z_label], bins))

    return redshift_bins, bin_data, bins



def create_minimum_obj_bins(df, num_per_bin, z_label):
    """ This function calculates the redshift bins for the chi-squared fitting
    routine. The minimum number of objects per bin set the redshift bins.

        Parameters:
            df : dataframe
            Dataframe containing the features

            num_per_bin : integer
            Maximum number of objects per bin

            z_label : string
            Redshift label name

        Returns:

            redshift_bins : groupby element
            This groupby elements contains all values from the model catalog
            grouped by the redshift bins.

            bin_data : array-like ([ number of bins ]x4)
            This array contains the information about the bins. It contains the
            lower redshift boundary, the upper redshift boundary, the mean
            redshift and the redshift width in this order.

            bins : array-like
            The bins calculated by this method

        """


    # Copy redshift columns from dataframe DataFrame
    redshifts = df[[z_label]]

    redshifts = redshifts.sort_values(z_label)

    # generate bin borders
    a = np.array_split(np.array(redshifts),redshifts.shape[0]/num_per_bin)

    bins =[0]

    # calculate lower bin border
    for i in range(len(a)-1):
      bins.append((a[i].max() + a[i+1].min())/2.)

    bins.append(a[len(a)-1].max())

    bins = np.array(bins)

    # calculating the middle redshift and bin width
    bin_middle = np.array((bins[:-1]+bins[1:])/2.)
    bin_width = np.array(bins[1:]-bins[:-1])

    # create array that holds all information about the bins
    # lower edge, upper edge, middle redshift, redshift width
    bin_data = np.array([bins[:-1],bins[1:],bin_middle,bin_width])

    # Generate pandas groups from which we can calculate the colors
    redshift_bins = df.groupby(pd.cut(df[z_label], bins))

    return redshift_bins, bin_data, bins



def build_star_model(star_catalog, features, label):
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

            label : column name with star classes

        Returns:
            star_model : Dataframe containing the median values of all the
            features as well as the information about the bins
        """


    # Generate pandas groups from which we can calculate the colors
    star_bins = star_catalog.groupby(star_catalog[label])

    # copy star class columns from star catalog to model DataFrame
    star_model = pd.DataFrame(star_bins.agg([np.median])['ug'].index,columns=[ \
                                                                label])

    # add number of objects per star class as a column
    star_model['class_counts'] = star_bins.size().values


    for i in range(len(features)):
      star_model[features[i]] = np.array(star_bins.agg([np.median])[features[i]])

    return star_model



def build_binned_model(redshift_bins, bin_data, features,
                                                    model_type = 'median'):
    """ This function calculates the median values for each bin and creates the
    feature_model which not only contains the feature median values but also the
    information about the bins from bin_data.

        Parameters:
            redshift_bins : groupby element
            This groupby elements contains all values from the model catalog
            grouped by the redshift bins.

            bin_data : array-like ([ number of bins ]x4)
            This array contains the information about the bins. It contains the
            lower redshift boundary, the upper redshift boundary, the mean
            redshift and the redshift width in this order.

            features : list of string
            Names of the features considered (flux_ratio_names, color_names)

            model_type : string
            The model_type parameter can be either 'median' or 'mean'. It determines
            how the model features are determined from the dataframe

        Returns:
            feature_model : Dataframe containing the median values of all the
            features as well as the bin_data information
        """

    # This function calculates the median feature value for all bins and stores
    #  it as the feature model. The feature model dataframe also consists of
    #  the bin_data

    feature_model = pd.DataFrame(bin_data.T,columns=['bin_lower', \
                                    'bin_upper','bin_middle','bin_width'])

    if model_type == 'median' :
        for i in range(len(features)):
          feature_model[features[i]] = np.array( \
                        redshift_bins.agg([np.median])[features[i]])

    elif model_type == 'mean' :
        for i in range(len(features)):
          feature_model[features[i]] = np.array( \
                        redshift_bins.agg([np.mean])[features[i]])
    else:
        print 'Model type not recognized, please choose either '
        print ' "median" or "mean". '



    return feature_model



def photoz_fit(df_train,df_pred,features, z_label, params):

    """This routine calculates the photometric redshift of quasars based on
    the features passed to it. It sets up a data model of the median features as
    a function of redshift and then calculates the chi-squared of each datum
    given the model.
    The best-fitting redshift is estimated using the largest area peak under the
    chi-squared probability distribution.

    Parameters:
            df_train : pandas dataframe
            The training set for the photometric redshift estimation

            df_pred : pandas dataframe
            The set to predict the photometric redshift of

            features : list of strings
            List of features, for this routine these are generally either
            colors or flux ratios.

            z_label : string
            The true redshift label of the dataframe

            params : dictionary
            List of input parameters for the various routines

            rand_state : integer
            Setting the random state variables to ensure reproducibility
    """

    # Create redshift bin structure
    if params['binning'] == "minimum":

        num_per_bin = params['bin_param']

        redshift_bins, bin_data, bins = create_minimum_obj_bins(df_train,
                                                        num_per_bin, z_label)
    elif params['binning'] == 'simple':

        bin_size = params['bin_param']
        # the bin size is only tested for a value of 0.025

        redshift_bins, bin_data, bins = create_simple_bins(df_train, bin_size, z_label)

    else :
        print 'Binning option not recognized'
        print ', please choose either "minimum" or "simple" '

    # Build feature model
    model_type = params['model_type']
    feature_model = build_binned_model(redshift_bins, bin_data,
                                                    features, model_type)

    # Calculate the chi-squared probability distribution
    pdf_array = pfz.calc_photoz_pdf_binned(feature_model, df_pred, features)

    # Determine the best-fitting redshift
    df_pred = pfz.find_max_prob(pdf_array, df_pred)
    df_pred = pfz.calc_prob_peaks(pdf_array, df_pred, bins)

    pdf_prob = pd.DataFrame(pdf_array[1],index=df_pred.index,columns=bin_data[2,:])
    pdf_chisq = pd.DataFrame(pdf_array[2],index=df_pred.index,columns=bin_data[2,:])

    return df_pred, pdf_prob, pdf_chisq



def star_fit(df_train, df_pred, features, class_label, params):

    """ This routine calculates the best fit stellar class for the predicition
    set features passed to it. It sets up a data model of the median features as
    a function of stellar class and then calculates the chi-squared of each
    datum given the model.
    The best-fitting class is simply the one with the best fit.

    Parameters:
            df_train : pandas dataframe
            The training set for the stellar classification

            df_pred : pandas dataframe
            The prediction set for stellar classification

            features : list of strings
            List of features, for this routine these are generally either
            colors or flux ratios.

            class_label : string
            The true stellar class label of the dataframe column

            params : dictionary
            List of input parameters for the various routines

            rand_state : integer
            Setting the random state variables to ensure reproducibility
    """

    # Build the model flux ratios from the training set
    star_model = build_star_model(df_train,features,class_label)

    # Calculate the chi-squared probability distribution

    pdf_array = pfsc.calc_star_class_pdf_binned(star_model,df_pred,
                                                            features)

    # Determine the stellar class
    df_pred = pfsc.find_max_prob_star(pdf_array,df_pred)

    pdf_prob = pd.DataFrame(pdf_array[1],index=df_pred.index,columns=star_model.star_class)
    pdf_chisq = pd.DataFrame(pdf_array[2],index=df_pred.index,columns=star_model.star_class)


    return df_pred, pdf_prob, pdf_chisq, star_model


def photoz_fit_test(df, features, z_label, params, rand_state, save_data=False,
    save_name = 'test'):

    """This tests the calculation of the estimated redshift of quasars based on
    the features passed to it. It sets up a data model of the median features as
    a function of redshift and then calculates the chi-squared of each datum
    given the model.
    The best-fitting redshift is estimated using the largest area peak under the
    chi-squared probability distribution.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            chi-squared fitting

            features : list of strings
            List of features, for this routine these are generally either
            colors or flux ratios.

            z_label : string
            The true redshift label of the dataframe

            params : dictionary
            List of input parameters for the various routines

            rand_state : integer
            Setting the random state variables to ensure reproducibility

            save_data : boolean (default = False)
            Boolean to determine whether data will be saved in the process

            save_name : string
            Name for the folder extension and the data file
    """

    # Buiding the test and training samples by random split
    df_train,df_test = train_test_split(df, test_size=0.2,
                                                    random_state=rand_state)

    df_test, pdf_prob, pdf_chisq = \
            photoz_fit(df_train,df_test,features, z_label, params)

    # Analyze the best-fitting redshift
    y_true = df_test[z_label].values
    y_pred = df_test.pf_photoz.values

    print r2_score(y_true, y_pred)

    pz_an.plot_redshifts(y_true,y_pred)
    pz_an.plot_error_hist(y_true,y_pred)

    plt.show()
    y_true = df_test[z_label].values
    y_pred = df_test.peak_a_mode.values

    print r2_score(y_true, y_pred)

    pz_an.plot_redshifts(y_true,y_pred)
    pz_an.plot_error_hist(y_true,y_pred)


    # Save results of the chi-squared fitting process if selected
    if save_data:
        work_dir = 'photoz_'+str(save_name)+'/'

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        pdf_prob.to_hdf(work_dir+'prob.hdf5','data')
        pdf_chisq.to_hdf(work_dir+'chisq.hdf5','data')
        df_test.to_hdf(work_dir+'df_test.hdf5','data')


    plt.show()


def star_fit_test(df, features, class_label, params, rand_state, save_data=False,
    save_name = 'test'):

    """ This routine tests the calculation of the star classification based on
    features passed to it. It sets up a data model of the median features as a
    function of stellar class and then calculates the chi-squared of each
    datum given the model.
    The best-fitting class is simply the one with the best fit.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            chi-sqaured fitting.

            features : list of strings
            List of features, for this routine these are generally either
            colors or flux ratios.

            class_label : string
            The true stellar class label of the dataframe column

            params : dictionary
            List of input parameters for the various routines

            rand_state : integer
            Setting the random state variables to ensure reproducibility

            save_data : boolean (default = False)
            Boolean to determine whether data will be saved in the process

            save_name : string
            Name for the folder extension and the data file

    """

    # Buiding the test and training samples by random split
    df_train,df_test = train_test_split(df, test_size=0.2,random_state=1)

    df_test, pdf_prob, pdf_chisq, star_model = \
                    star_fit(df_train, df_test, features, class_label, params)

    # Plot the confusion matrix for the test sample
    y_true = df_test[class_label]
    y_pred = df_test['pf_star_class']
    class_names = star_model[class_label].values

    cnf_matrix = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

    ml_an.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')

    plt.show()

    # Evaluate the fit using the f1 score

    print f1_score(y_true, y_pred, labels=class_names, average='weighted')

    for label in class_names:
        f1 =  f1_score(y_true, y_pred, labels=label, average='weighted')
        print label, f1



    if save_data:
        work_dir = 'starclass_'+str(save_name)+'/'

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        pdf_prob.to_hdf(work_dir+'star_prob.hdf5','data')
        pdf_chisq.to_hdf(work_dir+'star_chisq.hdf5','data')
        df_test.to_hdf(work_dir+'df_test.hdf5','data')
