import os
import pandas as pd
import matplotlib.pyplot as plt

import model_functions as mod
from colorfitting_find_photoz import *
from colorfitting_find_star_class import *
import colorfitting_analysis as cf_an


def test_star_classification(file_prefix, star_catalog,obj_catalog,flux_ratio_names):

    #specify work directory
    work_dir = file_prefix+'star_class/'

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    star_model = mod.build_star_model(star_catalog,flux_ratio_names)


    pdf_array = calc_star_class_pdf_binned(star_model,obj_catalog,flux_ratio_names)

    obj_catalog = find_max_prob_star(pdf_array,obj_catalog)
    obj_catalog.to_hdf(work_dir+'wise_tmass_sdss_bright_fluxes_starclass.hdf5','data')

    pdf = pd.DataFrame(pdf_array[1],index=obj_catalog.wise_designation,columns=star_model.star_class)
    pdf.to_csv(work_dir+'star_prob.csv')

    pdf = pd.DataFrame(pdf_array[2],index=obj_catalog.wise_designation,columns=star_model.star_class)
    pdf.to_csv(work_dir+'star_chisq.csv')

    obj_catalog.to_hdf(work_dir+'wise_tmass_sdss_bright_fluxes_starclass.hdf5','data')

    # cf_an.plot_star_classes(obj_catalog)
    # plt.show()



### TEST STAR MODEL STUFF

file_prefix = 'bright_catalog'

passband_names = ['SDSS_u',\
            'SDSS_g',\
            'SDSS_r',\
            'SDSS_i',\
            'SDSS_z',\
            '2MASS_j',\
            '2MASS_ks',\
            'WISE_w1',\
            'WISE_w2']

star_catalog = pd.read_csv('models/DR10_star_flux_cat.csv')

# star_catalog = star_catalog.sample(frac=1.0)

# Build flux ratios for the star_catalog
star_catalog, flux_ratio_names = mod.prepare_flux_ratio_catalog(star_catalog,passband_names, sigma=True)

# obj_catalog = star_catalog.sample(frac=0.1)

# print obj_catalog.columns

# star_catalog = star_catalog.drop(obj_catalog.index)

obj_catalog = pd.read_hdf('models/wise_tmass_sdss_bright_fluxes.hdf5','data')
obj_catalog, flux_ratio_names = mod.prepare_flux_ratio_catalog(obj_catalog,passband_names, sigma=True)
# cf_an.plot_star_classes(obj_catalog)
# plt.show()
print obj_catalog.columns

test_star_classification(file_prefix, star_catalog,obj_catalog,flux_ratio_names)
