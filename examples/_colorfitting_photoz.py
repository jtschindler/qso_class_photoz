import os
import pandas as pd
import matplotlib.pyplot as plt

import model_functions as mod
from colorfitting_find_photoz import *
import photoz_analysis as pz_an
from colorfitting_find_star_class import *


def test_min_bin(passband_names,objects_per_bin,model_catalog, obj_catalog, \
                                            file_prefix, flux_ratio_names):


    for num in objects_per_bin:

        #specify work directory
        work_dir = file_prefix+'min_bin_test_'+str(num)+'/'

        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        redshift_bins, bin_data = mod.create_minimum_obj_bins(model_catalog,num, 'redshift')

        flux_model = mod.build_binned_model(redshift_bins,bin_data,flux_ratio_names)


        pdf_array = calc_photoz_pdf_binned(flux_model,obj_catalog,flux_ratio_names)

        obj_catalog = find_max_prob(pdf_array,obj_catalog)

        pz_an.plot_error_hist(obj_catalog.redshift.values, obj_catalog.photo_z.values)
        plt.show()
        pz_an.plot_redshifts(obj_catalog.redshift.values, obj_catalog.photo_z.values)
        plt.show()

        pdf = pd.DataFrame(pdf_array[1],index=obj_catalog.designation,columns=bin_data[2,:])
        pdf.to_csv(work_dir+'prob.csv')

        pdf = pd.DataFrame(pdf_array[2],index=obj_catalog.designation,columns=bin_data[2,:])
        pdf.to_csv(work_dir+'chisq.csv')


        obj_catalog.to_csv(work_dir+'obj_catalog.csv')



def test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names):

    work_dir = file_prefix+'simple_bin_test_'+str(bin_size)+'/'

    if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    redshift_bins, bin_data = mod.create_simple_bins(model_catalog,bin_size)

    flux_model = mod.build_binned_model(redshift_bins,bin_data,flux_ratio_names)

    pdf_array = calc_photoz_pdf_binned(flux_model,obj_catalog,flux_ratio_names)

    obj_catalog = find_max_prob(pdf_array,obj_catalog)

    pz_an.plot_error_hist(obj_catalog.redshift.values, obj_catalog.photo_z.values)
    plt.show()
    pz_an.plot_redshifts(obj_catalog.redshift.values, obj_catalog.photo_z.vaues)
    plt.show()

    pdf = pd.DataFrame(pdf_array[1],index=obj_catalog.designation,columns=bin_data[2,:])
    pdf.to_csv(work_dir+'prob.csv')

    pdf = pd.DataFrame(pdf_array[2],index=obj_catalog.designation,columns=bin_data[2,:])
    pdf.to_csv(work_dir+'chisq.csv')

    obj_catalog.to_csv(work_dir+'obj_catalog.csv')



def test_photoz_SDSSWISE4():

    #specify model directory
    model_dir = 'models/'

    #specify model catalog filename
    # redshift model is based on this catalog (this can be a quasar catalog or a template catalog)
    model_catalog_filename = model_dir+'DR7_DR12Q_flux_cat.csv'

    #folder prefix
    file_prefix = 'SDSSWISE4'

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            'WISE_w1',\
            'WISE_w2',\
            'WISE_w3',\
            'WISE_w4']

    #-----------------------------------------------------------------------------
    # Build model and obj catalog from randomly drawing 50% of the model_catalog
    #-----------------------------------------------------------------------------


    model_catalog = pd.read_csv(model_catalog_filename)

    model_catalog,flux_ratio_names = mod.prepare_flux_ratio_catalog(model_catalog, \
    passband_names, sigma=True)

    obj_catalog = model_catalog.sample(frac=0.5)

    model_catalog = model_catalog.drop(obj_catalog.index)

    #-----------------------------------------------------------------------------
    # Test the minimum bin technique
    #-----------------------------------------------------------------------------
    objects_per_bin = [200,100,50]

    test_min_bin(passband_names,objects_per_bin,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    #-----------------------------------------------------------------------------
    # Test the simple binning technique
    #-----------------------------------------------------------------------------

    bin_size = 0.0125

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    bin_size = 0.025

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)




def test_photoz_all():

    #specify work directory
    work_dir = file_prefix+'min_bin_test_'+str(num)+'/'

    if not os.path.exists(work_dir):
            os.makedirs(work_dir)

    #specify model directory
    model_dir = 'models/'

    #specify model catalog filename
    # redshift model is based on this catalog (this can be a quasar catalog or a template catalog)
    model_catalog_filename = model_dir+'DR7_DR12Q_flux_cat.csv'

    #folder prefix
    file_prefix = 'SDSS2MASSWISE4'

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            '2MASS_j',\
            '2MASS_h',\
            '2MASS_ks',\
            'WISE_w1',\
            'WISE_w2',\
            'WISE_w3',\
            'WISE_w4']

    #-----------------------------------------------------------------------------
    # Build model and obj catalog from randomly drawing 50% of the model_catalog
    #-----------------------------------------------------------------------------


    model_catalog = pd.read_csv(model_catalog_filename)

    model_catalog,flux_ratio_names = mod.prepare_flux_ratio_catalog(model_catalog, \
    passband_names, sigma=True)

    obj_catalog = model_catalog.sample(frac=0.5)

    model_catalog = model_catalog.drop(obj_catalog.index)

    #-----------------------------------------------------------------------------
    # Test the minimum bin technique
    #-----------------------------------------------------------------------------
    objects_per_bin = [200,100,50]

    test_min_bin(passband_names,objects_per_bin,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    #-----------------------------------------------------------------------------
    # Test the simple binning technique
    #-----------------------------------------------------------------------------

    bin_size = 0.0125

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    bin_size = 0.025

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)


def test_photoz_SDSS2MASS():

    #specify model directory
    model_dir = 'models/'

    #specify model catalog filename
    # redshift model is based on this catalog (this can be a quasar catalog or a template catalog)
    model_catalog_filename = model_dir+'DR7_DR12Q_flux_cat.csv'

    #folder prefix
    file_prefix = 'SDSS2MASS'

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            '2MASS_j',\
            '2MASS_h',\
            '2MASS_ks']

    #-----------------------------------------------------------------------------
    # Build model and obj catalog from randomly drawing 50% of the model_catalog
    #-----------------------------------------------------------------------------


    model_catalog = pd.read_csv(model_catalog_filename)

    model_catalog,flux_ratio_names = mod.prepare_flux_ratio_catalog(model_catalog, \
    passband_names, sigma=True)

    obj_catalog = model_catalog.sample(frac=0.5)

    model_catalog = model_catalog.drop(obj_catalog.index)

    #-----------------------------------------------------------------------------
    # Test the minimum bin technique
    #-----------------------------------------------------------------------------
    objects_per_bin = [200,100,50]

    test_min_bin(passband_names,objects_per_bin,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    #-----------------------------------------------------------------------------
    # Test the simple binning technique
    #-----------------------------------------------------------------------------

    bin_size = 0.0125

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    bin_size = 0.025

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)




def test_photoz_2MASSWISE4():

    #specify model directory
    model_dir = 'models/'

    #specify model catalog filename
    # redshift model is based on this catalog (this can be a quasar catalog or a template catalog)
    model_catalog_filename = model_dir+'DR7_DR12Q_flux_cat.csv'

    #folder prefix
    file_prefix = '2MASSWISE4'

    #specify passband and other column names for model file
    passband_names = ['2MASS_j',\
            '2MASS_h',\
            '2MASS_ks',\
            'WISE_w1',\
            'WISE_w2',\
            'WISE_w3',\
            'WISE_w4']

    #-----------------------------------------------------------------------------
    # Build model and obj catalog from randomly drawing 50% of the model_catalog
    #-----------------------------------------------------------------------------


    model_catalog = pd.read_csv(model_catalog_filename)

    model_catalog,flux_ratio_names = mod.prepare_flux_ratio_catalog(model_catalog, \
    passband_names, sigma=True)

    obj_catalog = model_catalog.sample(frac=0.5)

    model_catalog = model_catalog.drop(obj_catalog.index)

    #-----------------------------------------------------------------------------
    # Test the minimum bin technique
    #-----------------------------------------------------------------------------
    objects_per_bin = [200,100,50]

    test_min_bin(passband_names,objects_per_bin,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    #-----------------------------------------------------------------------------
    # Test the simple binning technique
    #-----------------------------------------------------------------------------

    bin_size = 0.0125

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    bin_size = 0.025

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)



def test_photoz_SDSS2MASSWISE2():

    #specify model directory
    model_dir = 'models/'

    #specify model catalog filename
    # redshift model is based on this catalog (this can be a quasar catalog or a template catalog)
    model_catalog_filename = model_dir+'DR7_DR12Q_flux_cat.csv'

    #folder prefix
    file_prefix = 'SDSS2MASSWISE2'

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            '2MASS_j',\
            '2MASS_h',\
            '2MASS_ks',\
            'WISE_w1',\
            'WISE_w2']

    #-----------------------------------------------------------------------------
    # Build model and obj catalog from randomly drawing 50% of the model_catalog
    #-----------------------------------------------------------------------------


    model_catalog = pd.read_csv(model_catalog_filename)

    model_catalog,flux_ratio_names = mod.prepare_flux_ratio_catalog(model_catalog, \
    passband_names, sigma=True)

    obj_catalog = model_catalog.sample(frac=0.5)

    model_catalog = model_catalog.drop(obj_catalog.index)

    #-----------------------------------------------------------------------------
    # Test the minimum bin technique
    #-----------------------------------------------------------------------------
    objects_per_bin = [200,100,50]

    test_min_bin(passband_names,objects_per_bin,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    #-----------------------------------------------------------------------------
    # Test the simple binning technique
    #-----------------------------------------------------------------------------

    bin_size = 0.0125

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)

    bin_size = 0.025

    test_simple_bin(passband_names,bin_size,model_catalog, obj_catalog, file_prefix, flux_ratio_names)






def test_mixed_class():

    #specify model directory
    model_dir = 'models/'

    #specify model catalog filename
    # redshift model is based on this catalog (this can be a quasar catalog or a template catalog)
    qso_catalog_filename = model_dir+'DR7_DR12Q_flux_cat.csv'
    # star model is based on this catalog (default is SDSS stellar catalog DR10)
    star_catalog_filename = model_dir+'DR10_star_flux_cat.csv'

    #folder prefix
    file_prefix = ''

    #specify work directory
    work_dir = file_prefix+'bright_catalog/'

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    #specify passband and other column names for model file
    passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            '2MASS_j',\
            '2MASS_h',\
            '2MASS_ks',\
            'WISE_w1',\
            'WISE_w2']


    #-----------------------------------------------------------------------------
    # Build model and obj catalogs using the generate_mixed_sample() routine
    # This automatically builds the subsamples for the catalogs and
    # removes all samples stars and quasars from their respective parent catalog
    #-----------------------------------------------------------------------------

    star_cat = pd.read_csv(star_catalog_filename)

    qso_cat = pd.read_csv(qso_catalog_filename)

    # Build flux ratios
    qso_cat,flux_ratio_names = mod.prepare_flux_ratio_catalog(qso_cat, \
    passband_names, sigma=True)
    star_cat,flux_ratio_names = mod.prepare_flux_ratio_catalog(star_cat, \
    passband_names, sigma=True)

    # qso_test = qso_cat.sample(frac = 0.02)
    # qso_cat.drop(qso_test.index, inplace=True)
    # star_test = star_cat.sample(frac = 0.02)
    # star_cat.drop(star_test.index, inplace=True)
    #
    # obj_cat = mod.build_full_sample(star_test,qso_test,50, return_cats=False)

    obj_cat = pd.read_hdf('models/wise_tmass_sdss_bright_fluxes.hdf5')
    obj_cat ,flux_ratio_names = mod.prepare_flux_ratio_catalog(obj_cat, \
    passband_names, sigma=True)

    #-----------------------------------------------------------------------------
    # Build the flux model catalogs for the quasar and star training sets
    #-----------------------------------------------------------------------------

    #Quasars

    redshift_bins, bin_data = mod.create_minimum_obj_bins(qso_cat,50, 'redshift')

    qso_model = mod.build_binned_model(redshift_bins,bin_data,flux_ratio_names)


    #Stars

    star_model = mod.build_star_model(star_cat,flux_ratio_names)

    #-----------------------------------------------------------------------------
    # Fit all objects in the obj_cat to the star and quasar models
    #-----------------------------------------------------------------------------

    #Fit to Quasar model

    pdf_array = calc_photoz_pdf_binned(qso_model,obj_cat,flux_ratio_names)

    obj_cat = find_max_prob(pdf_array,obj_cat)

    # print obj_cat.redshift.values
    # print obj_cat.photo_z.values

    # qso_obj = obj_cat.query('redshift > 0.0')

    # pz_an.plot_error_hist(qso_obj.redshift.values, qso_obj.photo_z.values)
    # plt.show()
    # pz_an.plot_redshifts(qso_obj.redshift.values, qso_obj.photo_z.values)
    # plt.show()
    pdf = pd.DataFrame(pdf_array[1],index=obj_cat.wise_designation,columns=bin_data[2,:])
    pdf.to_csv(work_dir+'qso_prob.csv')

    pdf = pd.DataFrame(pdf_array[2],index=obj_cat.wise_designation,columns=bin_data[2,:])
    pdf.to_csv(work_dir+'qso_redchisq.csv')


    #Fit to Star model

    pdf_array = calc_star_class_pdf_binned(star_model,obj_cat,flux_ratio_names)

    obj_cat = find_max_prob_star(pdf_array,obj_cat)

    pdf = pd.DataFrame(pdf_array[1],index=obj_cat.wise_designation,columns=star_model.star_class)
    pdf.to_csv(work_dir+'star_prob.csv')

    pdf = pd.DataFrame(pdf_array[2],index=obj_cat.wise_designation,columns=star_model.star_class)
    pdf.to_csv(work_dir+'star_redchisq.csv')

    obj_cat.to_hdf(work_dir+'wise_tmass_sdss_bright_cf_pred.hdf5','data')

    return 1


test_mixed_class()
