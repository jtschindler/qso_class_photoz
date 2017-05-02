import numpy as np
from time import gmtime, strftime




def calc_photoz_pdf_binned(feature_model, df_test, features):
    """This function calculates the chi^2 values and probabilities for each
    entry in the feature_model to each object in the df_test and saves them in
    a list of arrays.


    Parameters:
        feature_model : array-like, shape (n_samples)
        Array containing the true values of the regression

        df_test : array-like, shape (n_samples)
        Array containing the objects which flux ratios are fit against the flux
        model to evaluate the photometric redshift.

        features : list of strings
        Names of the features to consider, for this routine these are generally
        colors or flux ratios

    Returns:
        pdf_array : list of arrays [bin_data,[ #df_test x #feature_models] x2 ]
        The first object in this list of arrays holds information about the bin
        sizes, whereas the second and third object hold two matrices with
        #df_test x #feature_models entries, where the first holds the chi^2
        value and the second holds the probability.

    """

    # PARALLELIZE LATER ON

    # Generate the structure for the probability distribution (pd)
    num_objs =  df_test.shape[0]
    num_models =  feature_model.shape[0]

    arr = np.zeros([num_objs,num_models])
    arr2 = np.zeros([num_objs,num_models])

    # The probability density function array consists of two parts
    # The first object in the list consists of the bin_data
    # The second object in the list consists of the all pds

    pdf_array = [np.array(feature_model[['bin_middle','bin_width']]),arr,arr2]

    # Calculating the probabilities
    # obj_fr = object flux ratios
    # obj_fr_err = error on object flux ratios
    # model_fr = model flux ratios

    flux_ratio_err_names = []
    for name in features:
        flux_ratio_err_names.append('sigma_'+name)

    for i in range(len(df_test.index)):
        obj_fr = np.array(df_test.loc[df_test.index[i],features])
        obj_fr_err = np.array(df_test.loc[df_test.index[i],flux_ratio_err_names])

        if i%100 == 0:
            print '{0:.1f}'.format(float(i)/num_objs*100.)+'% '\
                                        +strftime("%Y-%m-%d %H:%M:%S", gmtime())

        for j in range(len(feature_model.index)):

            model_fr = np.array(feature_model.loc[feature_model.index[j],features])

            #calculate the chi_squared
            chi_squared = np.sum((obj_fr-model_fr)**2/obj_fr_err**2)

            #calculate the probability
            prob = np.exp(-chi_squared/2.)/np.sqrt(np.sum(obj_fr_err**2))

            pdf_array[2][i,j] = chi_squared/len(features) #save reduced chi squared
            pdf_array[1][i,j] = prob


    return pdf_array


def find_max_prob(pdf_array,df_test):
    """This function finds the most probable redshift for each object in the
    df_test and adds columns containing it and additional data from the
    flux ratio fitting process.


    Parameters:
        pdf_array : list of arrays [bin_data,[ #df_test x #feature_models] x2 ]
        The first object in this list of arrays holds information about the bin
        sizes, whereas the second and third object hold two matrices with
        #df_test x #feature_models entries, where the first holds the chi^2
        value and the second holds the probability.

        df_test : array-like, shape (n_samples)
        Array containing the objects which flux ratios were fit against the flux
        model to evaluate the photometric redshift.

    Returns:

        df_test : array-like, shape (n_samples)
        Array containing the objects which flux ratios were fit against the flux
        model to evaluate the photometric redshift. Additional columns are added
        which contain the most probable photometric redshift as well as
        information about the chi^2 value, the probability and the photometric
        redshift bin it was evaluated in.

    """

    index = []
    prob = []
    chi_squared = []
    for i in range(len(pdf_array[1][:,0])):

        max_index = max(xrange(len(pdf_array[1][i,:])),key=pdf_array[1][i,:].__getitem__)
        index.append(max_index)
        prob.append(pdf_array[1][i,max_index])
        chi_squared.append(pdf_array[2][i,max_index])


    # The first object in the pdf_array list [0]
    # stores the redshift bins
    df_test.loc[:,'pf_photoz'] = pdf_array[0][index,0]
    df_test.loc[:,'pf_qso_redchisq'] = chi_squared
    df_test.loc[:,'pf_qso_prob'] = prob
    df_test.loc[:,'pf_z_width'] = pdf_array[0][index,1]

    return df_test



def calc_prob_peaks(pdf_array,df_test,bins):

    """This function calculates the boundaries, probabilities and mode redshifts
    of the two largest peaks in the redshift probability distribution.
    In order to do this the pdf is normalized and then the uniform pdf is
    subtracted to find all positive peaks. The data of the two highest of those
    peaks is then saved and returned in the df_test dataframe

    CURRENTLY THE REDSHIFT SAVED IS THE MODE AND THEREFORE SIMILAR TO
    find_max_prob. For the future calculating the median redshift of the peak
    would be the better thing to do.


    Parameters:
        pdf_array : list of arrays [bin_data,[ #df_test x #feature_models] x2 ]
        The first object in this list of arrays holds information about the bin
        sizes, whereas the second and third object hold two matrices with
        #df_test x #feature_models entries, where the first holds the chi^2
        value and the second holds the probability.

        df_test : array-like, shape (n_samples)
        Array containing the objects which flux ratios were fit against the flux
        model to evaluate the photometric redshift.

        bins : array-like
        The bins calculated by the binning method before

    Returns:

        df_test : array-like, shape (n_samples)
        Array containing the objects which flux ratios were fit against the flux
        model to evaluate the photometric redshift. Additional columns are added
        which contain the most probable photometric redshift as well as
        information about the chi^2 value, the probability and the photometric
        redshift bin it was evaluated in.

    """


    bin_middle = pdf_array[0][:,0]
    bin_width  = pdf_array[0][:,1]

    peak_a_left = []
    peak_b_left = []
    peak_a_right = []
    peak_b_right = []
    peak_a_prob = []
    peak_b_prob = []
    peak_a_mode = []
    peak_b_mode = []


    for kdx in range(len(pdf_array[1][:,0])):
        pdf = pdf_array[1][kdx,:]

        # Normalize probability to one
        Area = np.sum(bin_width * pdf)
        pdf = pdf / Area

        # Subtract uniform probability distribution
        # print 1.0/(max(bins)-min(bins))
        uni = 1.0/(max(bins)-min(bins)) + pdf * 0
        pdf = pdf - uni



        # Find peaks and calculate their integrated probability and width

        # Set up the containers to store the data of the two largest peaks
        # left boundary, right boundary, peak probability, mode
        peak_a = (0,0,0,0)
        peak_b = (0,0,0,0)

        # Set up idx iterator
        idx = 0

        # Loop over all values of the pdf
        while idx < (len(pdf)):

            # If the pdf value is positive :
            if pdf[idx] > 0:

                # Save left boundary of peak
                p_left = bins[idx]
                p_right = bins[idx+1]
                # Set up Area of peak with first entry
                A = pdf[idx] * bin_width[idx]

                # Increase the iterator
                jdx = idx + 1



                # If idx was not the last cell
                while jdx < len(pdf):
                    # If the value in the jdx (firs: idx+1) cell is positive
                    if pdf[jdx] > 0  :

                        # For each positive value sum the area under the peak
                        A += pdf[jdx] * bin_width[jdx]
                        # For each positive value update the right boundary
                        p_right = bins[jdx+1]
                        # Increase iterator
                        jdx += 1
                    # If the value in the jdx (firs: idx+1) cell is not positive
                    else :

                        # Save peak values
                        max_index = max(xrange(len(pdf[idx:jdx])),
                                                key=pdf[idx:jdx].__getitem__)

                        mode = bin_middle[idx:jdx][max_index]


                        peak = (p_left,p_right,A,mode)

                        if peak[2] > peak_b[2]:

                            if peak[2] > peak_a[2]:
                                peak_b = peak_a
                                peak_a = peak
                            else:
                                peak_b = peak

                        # Set idx to first value after the peak
                        idx = jdx
                        # Set jdx to max value to end the while statement
                        jdx = len(pdf)


                # If idx was the last cell
                if jdx == len(pdf):

                    # mode = bin_middle[idx]
                    #
                    # peak = (p_left,p_right,A,mode)
                    #
                    # if peak[2] > peak_b[2]:
                    #
                    #     if peak[2] > peak_a[2]:
                    #         peak_b = peak_a
                    #         peak_a = peak
                    #     else:
                    #         peak_b = peak

                    idx += 1


            else :
                # Increase iterator
                idx += 1

        peak_a_left.append(peak_a[0])
        peak_a_right.append(peak_a[1])
        peak_a_prob.append(peak_a[2])
        peak_a_mode.append(peak_a[3])
        peak_b_left.append(peak_b[0])
        peak_b_right.append(peak_b[1])
        peak_b_prob.append(peak_b[2])
        peak_b_mode.append(peak_b[3])

    print "DONE PEAK CALCULATION"

    df_test.loc[:,'peak_a_left'] = peak_a_left
    df_test.loc[:,'peak_a_right'] = peak_a_right
    df_test.loc[:,'peak_a_prob'] = peak_a_prob
    df_test.loc[:,'peak_a_mode'] = peak_a_mode
    df_test.loc[:,'peak_b_left'] = peak_b_left
    df_test.loc[:,'peak_b_right'] = peak_b_right
    df_test.loc[:,'peak_b_prob'] = peak_b_prob
    df_test.loc[:,'peak_b_mode'] = peak_b_mode

    return df_test



# def calc_photoz_pdf_spline(spline_list,df_test,features,z_resolution):
#     # Calculates the pdf for each obj in the df_test by
#     # using the chi_squared method
#
#     #generate the pdf structure
#     num_objs =  df_test.shape[0]
#
#     model_z = np.arange(0,5,z_resolution)
#
#     model_width = np.array([z_resolution]*model_z.shape[0])
#
#     num_models = model_z.shape[0]
#
#     arr = np.zeros([num_objs,num_models])
#     arr2 = np.zeros([num_objs,num_models])
#
#     # The probability density function array consists of two parts
#     # The first object in the list consists of the redshift values
#     # The second object in the list consists of the all pdfs
#     pdf_array = [np.array([model_z,model_width]).T,arr,arr2]
#
#     # Calculating the probabilities
#     # obj_fr = object flux ratios
#     # obj_fr_err = error on object flux ratios
#     # model_fr = model flux ratios
#
#     flux_ratio_err_names = []
#     for name in features:
#         flux_ratio_err_names.append('sigma_'+name)
#
#     model_array = np.zeros([len(features),model_z.shape[0]])
#
#     for i in range(len(features)):
#         model_array[i,:] = spline_list[i](model_z)
#
#     for i in range(len(df_test.index)):
#         obj_fr = np.array(df_test.loc[df_test.index[i],features])
#         obj_fr_err = np.array(df_test.loc[df_test.index[i],flux_ratio_err_names])
#
#         if i%100 == 0:
#            print '{0:.1f}'.format(float(i)/num_objs*100.)+'% '+strftime("%Y-%m-%d %H:%M:%S", gmtime())
#
#         for j in range(model_z.shape[0]):
#
#             model_fr = model_array[:,j]
#
#             #calculate the chi_squared
#             chi_squared = np.sum((obj_fr-model_fr)**2/obj_fr_err**2)
#             #calculate the probability
#             prob = np.exp(-chi_squared/2.)/np.sqrt(np.sum(obj_fr_err**2))
#
#             pdf_array[2][i,j] = chi_squared/len(features) #save reduced chi squared
#             pdf_array[1][i,j] = prob
#
#     return pdf_array
