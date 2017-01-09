import numpy as np
from time import gmtime, strftime



def calc_photoz_pdf_binned(flux_model,obj_catalog,flux_ratio_names):
    """This function calculates the chi^2 values and probabilities for each
    entry in the flux_model to each object in the obj_catalog and saves them in
    a list of arrays.


    Parameters:
        flux_model : array-like, shape (n_samples)
        Array containing the true values of the regression

        obj_catalog : array-like, shape (n_samples)
        Array containing the objects which flux ratios are fit against the flux
        model to evaluate the photometric redshift.

        flux_ratio_names : list of strings
        Names of the flux ratios to consider

    Returns:
        pdf_array : list of arrays [bin_data,[ #obj_catalog x #flux_models] x2 ]
        The first object in this list of arrays holds information about the bin
        sizes, whereas the second and third object hold two matrices with
        #obj_catalog x #flux_models entries, where the first holds the chi^2
        value and the second holds the probability.

    """


    # Calculates the pdf for each obj in the obj_catalog by
    # using the chi_squared method

    #generate the pdf structure
    num_objs =  obj_catalog.shape[0]
    num_models =  flux_model.shape[0]

    arr = np.zeros([num_objs,num_models])
    arr2 = np.zeros([num_objs,num_models])

    # The probability density function array consists of two parts
    # The first object in the list consists of the bin_data
    # The second object in the list consists of the all pdfs

    pdf_array = [np.array(flux_model[['bin_middle','bin_width']]),arr,arr2]

    # Calculating the probabilities
    # obj_fr = object flux ratios
    # obj_fr_err = error on object flux ratios
    # model_fr = model flux ratios

    flux_ratio_err_names = []
    for name in flux_ratio_names:
        flux_ratio_err_names.append('sigma_'+name)

    for i in range(len(obj_catalog.index)):
        obj_fr = np.array(obj_catalog.loc[obj_catalog.index[i],flux_ratio_names])
        obj_fr_err = np.array(obj_catalog.loc[obj_catalog.index[i],flux_ratio_err_names])

        if i%100 == 0:
            print '{0:.1f}'.format(float(i)/num_objs*100.)+'% '\
                                        +strftime("%Y-%m-%d %H:%M:%S", gmtime())

        for j in range(len(flux_model.index)):

            model_fr = np.array(flux_model.loc[flux_model.index[j],flux_ratio_names])

            #calculate the chi_squared
            chi_squared = np.sum((obj_fr-model_fr)**2/obj_fr_err**2)

            #calculate the probability
            prob = np.exp(-chi_squared/2.)/np.sqrt(np.sum(obj_fr_err**2))

            pdf_array[2][i,j] = chi_squared/len(flux_ratio_names) #save reduced chi squared
            pdf_array[1][i,j] = prob

    return pdf_array


def find_max_prob(pdf_array,object_catalog):
    """This function finds the most probable redshift for each object in the
    object_catalog and adds columns containing it and additional data from the
    flux ratio fitting process.


    Parameters:
        pdf_array : list of arrays [bin_data,[ #obj_catalog x #flux_models] x2 ]
        The first object in this list of arrays holds information about the bin
        sizes, whereas the second and third object hold two matrices with
        #obj_catalog x #flux_models entries, where the first holds the chi^2
        value and the second holds the probability.

        obj_catalog : array-like, shape (n_samples)
        Array containing the objects which flux ratios were fit against the flux
        model to evaluate the photometric redshift.

    Returns:

        obj_catalog : array-like, shape (n_samples)
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
    object_catalog['pf_photoz'] = pdf_array[0][index,0]
    object_catalog['pf_qso_redchisq'] = chi_squared
    object_catalog['pf_qso_prob'] = prob
    object_catalog['pf_z_width'] = pdf_array[0][index,1]

    return object_catalog



# def calc_photoz_pdf_spline(spline_list,obj_catalog,flux_ratio_names,z_resolution):
#     # Calculates the pdf for each obj in the obj_catalog by
#     # using the chi_squared method
#
#     #generate the pdf structure
#     num_objs =  obj_catalog.shape[0]
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
#     for name in flux_ratio_names:
#         flux_ratio_err_names.append('sigma_'+name)
#
#     model_array = np.zeros([len(flux_ratio_names),model_z.shape[0]])
#
#     for i in range(len(flux_ratio_names)):
#         model_array[i,:] = spline_list[i](model_z)
#
#     for i in range(len(obj_catalog.index)):
#         obj_fr = np.array(obj_catalog.loc[obj_catalog.index[i],flux_ratio_names])
#         obj_fr_err = np.array(obj_catalog.loc[obj_catalog.index[i],flux_ratio_err_names])
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
#             pdf_array[2][i,j] = chi_squared/len(flux_ratio_names) #save reduced chi squared
#             pdf_array[1][i,j] = prob
#
#     return pdf_array
