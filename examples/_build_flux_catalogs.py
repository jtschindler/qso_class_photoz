import pandas as pd
import numpy as np
import photometric_functions as phot

def build_flux_model_catalog_from_SDSS_QSO_cat(quasar_catalog_filename):

  try:
    quasar_catalog = pd.read_csv(quasar_catalog_filename)
  except:
    print "ERROR: Quasar catalog could not be read in. \n" \
      + "The Quasar catalog has to be in CSV format."

  #-----------------------------------------------------------------------------
  # Building the flux catalog DataFrame from the Quasar catalog
  #-----------------------------------------------------------------------------

  #specifiec columns will be copied and renames, additional columns may be added by the user here
  general_column_names = ['SDSS_NAME','RA','DEC','Z_VI']

  flux_catalog = quasar_catalog[general_column_names].copy()

  flux_catalog.rename(columns={"SDSS_NAME": "designation", "RA": "ra","DEC":"dec","Z_VI":"redshift"},inplace=True)


  #-----------------------------------------------------------------------------
  #  Specify fluxes and magnitudes from the quasar catalog to save in the new
  #  flux catalog
  #-----------------------------------------------------------------------------

  sdss_flux_names = ['PSFFLUX_U','PSFFLUX_G','PSFFLUX_R','PSFFLUX_I','PSFFLUX_Z']
  # fluxes in the SDSS bands u,g,r,i,z in nanomaggies
  sdss_flux_err_names = ['IVAR_PSFFLUX_U','IVAR_PSFFLUX_G','IVAR_PSFFLUX_R','IVAR_PSFFLUX_I','IVAR_PSFFLUX_Z',]
  # inverse variances of the fluxes in the u,g,r,i,z, bands in nanomaggies^-2

  #these are the column names for the SDSS fluxes in the output flux catalog
  sdss_bandpass_names =  ['SDSS_u',\
		  'SDSS_g',\
		  'SDSS_r',\
		  'SDSS_i',\
		  'SDSS_z']


  extinction_names = ['EXTINCTION_RECAL_U','EXTINCTION_RECAL_G','EXTINCTION_RECAL_R','EXTINCTION_RECAL_I','EXTINCTION_RECAL_Z']
  # Galactic extinction values in magnitudes for the SDSS bandpasses from Schlafly & Finkbeiner (2011)



  vega_mag_names =  ['JMAG','HMAG','KMAG','W1MAG','W2MAG','W3MAG','W4MAG']
  # magnitudes of other survey bandpasses in VEGA magnitudes
  vega_mag_err_names = ['ERR_JMAG','ERR_HMAG','ERR_KMAG', 'ERR_W1MAG','ERR_W2MAG','ERR_W3MAG','ERR_W4MAG']
  # 1-sigma error on magnitudes of other survey bandpasses in VEGA magnitudes

  # These are the column names for the other magnitudes fluxes in the output flux catalog
  # These names have to be in the corresponding order to the vega_mag_names above
  vega_bandpass_names = [ '2MASS_j',\
			   '2MASS_h',\
			   '2MASS_ks',\
		           'WISE_w1',\
		           'WISE_w2',\
		           'WISE_w3',\
		           'WISE_w4']

  #-----------------------------------------------------------------------------
  # Adjust SDSS fluxes using the correct AB magnitude zero point and deredden
  # them for the flux catalog
  #-----------------------------------------------------------------------------

  # Conversion from nanomaggies to Jansky and correction to correct zero point flux
  for i in range(len(sdss_bandpass_names)):
      flux_catalog[sdss_bandpass_names[i]] = quasar_catalog[sdss_flux_names[i]]*1e-9*3631*phot.VEGAtoAB_flux(sdss_bandpass_names[i])



  # Dereddening of the fluxes using the extinction values in the u passband
  for i in range(len(sdss_bandpass_names)):
      flux_catalog[sdss_bandpass_names[i]] = phot.deredden_flux(flux_catalog[sdss_bandpass_names[i]],sdss_bandpass_names[i],quasar_catalog.EXTINCTION_RECAL_U,'SDSS_u')

  #-----------------------------------------------------------------------------
  # Conversion of the Vega magnitudes from the other survey bandpasses to
  # fluxes, AB correctiond and reddening is applied before
  #-----------------------------------------------------------------------------

  for i in range(len(vega_bandpass_names)):
      # take care of no entries in the original catalog and replace the "0"s with NANs

      # ADD -9999 VALUES to NAN list
      flux_catalog[vega_bandpass_names[i]] = quasar_catalog[vega_mag_names[i]].replace(0.0,np.NaN)

      # convert VEGA to AB magnitudes
      flux_catalog[vega_bandpass_names[i]] = phot.VEGAtoAB(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i])

      # apply the correct dereddening
      flux_catalog[vega_bandpass_names[i]] = phot.deredden_mag(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i],quasar_catalog.EXTINCTION_RECAL_U,'SDSS_u')

      # convert exinction corrected AB magnitudes to fluxes in Jy
      flux_catalog[vega_bandpass_names[i]] = phot.ABMAGtoFLUX(flux_catalog[vega_bandpass_names[i]])

  #-----------------------------------------------------------------------------
  # Conversion of SDSS flux inverse variances into 1-sigma uncertainties
  #-----------------------------------------------------------------------------

  # The 1-sigma uncertainty for the dereddened flux is calculated by using the relative
  # 1#sigma uncertainty on the uncorrected flux
  for i in range(len(sdss_bandpass_names)):

      #calculate relative 1-sigma flux error
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = 1./np.sqrt(quasar_catalog[sdss_flux_err_names[i]])/quasar_catalog[sdss_flux_names[i]]

      #calculate absolute 1-sigma flux error on the extinction corrected flux value
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = flux_catalog['sigma_'+sdss_bandpass_names[i]] * flux_catalog[sdss_bandpass_names[i]]


  #print flux_cat.sigma_g/flux_cat.g
  #print  ( 1./np.sqrt(quasar_cat.IVAR_PSFFLUX_G)) / quasar_cat.PSFFLUX_G


  #-----------------------------------------------------------------------------
  # Conversion of VEGA magnitudes 1-sigma uncertainties to 1-sigma
  # uncertainties on the extinction corrected fluxes
  #-----------------------------------------------------------------------------


  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes
  for i in range(len(vega_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]] = quasar_catalog[vega_mag_err_names[i]].replace(0.0,np.NaN)

      # calculating the lower and upper magnitude levels
      low_mag = np.array(quasar_catalog[vega_mag_names[i]] + flux_catalog['sigma_'+vega_bandpass_names[i]])
      up_mag =  np.array(quasar_catalog[vega_mag_names[i]] - flux_catalog['sigma_'+vega_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,vega_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,vega_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,vega_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_RECAL_U),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,vega_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_RECAL_U),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ABMAGtoFLUX(low_mag)
      up_flux = phot.ABMAGtoFLUX(up_mag)

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+vega_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)


  print flux_catalog.columns

  flux_catalog.to_csv('models/DR12Q_flux_cat.csv')


















def build_flux_model_catalog_from_joint_SDSS_QSO_cat(quasar_catalog_filename):

  try:
    quasar_catalog = pd.read_csv(quasar_catalog_filename)
  except:
    print "ERROR: Quasar catalog could not be read in. \n" \
      + "The Quasar catalog has to be in CSV format."

  #-----------------------------------------------------------------------------
  # Building the flux catalog DataFrame from the Quasar catalog
  #-----------------------------------------------------------------------------

  #specifiec columns will be copied and renames, additional columns may be added by the user here
  general_column_names = ['NAME','RA','DEC','REDSHIFT', \
      'PSFMAG_U','PSFMAG_G','PSFMAG_R','PSFMAG_I','PSFMAG_Z', \
         'J_M_2MASS','H_M_2MASS','K_M_2MASS','W1MPRO','W2MPRO','W3MPRO','W4MPRO',\
             'PSFMAGERR_U','PSFMAGERR_G','PSFMAGERR_R','PSFMAGERR_I','PSFMAGERR_Z', \
         'J_MSIG_2MASS','H_MSIG_2MASS','K_MSIG_2MASS', 'W1SIGMPRO','W2SIGMPRO','W3SIGMPRO','W4SIGMPRO']

  flux_catalog = quasar_catalog[general_column_names].copy()

  flux_catalog.rename(columns={"NAME": "designation", "RA": "ra","DEC":"dec","REDSHIFT":"redshift"},inplace=True)


  #-----------------------------------------------------------------------------
  #  Specify magnitudes from the quasar catalog to save in the new
  #  flux catalog
  #-----------------------------------------------------------------------------

  sdss_mag_names = ['PSFMAG_U','PSFMAG_G','PSFMAG_R','PSFMAG_I','PSFMAG_Z']
  # magnitudes in the SDSS bands u,g,r,i,z in asinh mags
  sdss_mag_err_names = ['PSFMAGERR_U','PSFMAGERR_G','PSFMAGERR_R','PSFMAGERR_I','PSFMAGERR_Z',]
  # errors on the magnitudes in the u,g,r,i,z, bands in asinh mags

  #these are the column names for the SDSS fluxes in the output flux catalog
  sdss_bandpass_names =  ['SDSS_u',\
		  'SDSS_g',\
		  'SDSS_r',\
		  'SDSS_i',\
		  'SDSS_z']


  extinction_name = ['EXTINCTION_U']
  # Galactic extinction values in magnitudes for the SDSS bandpasses



  vega_mag_names =  ['J_M_2MASS','H_M_2MASS','K_M_2MASS','W1MPRO','W2MPRO','W3MPRO','W4MPRO']
  # magnitudes of other survey bandpasses in VEGA magnitudes
  vega_mag_err_names = ['J_MSIG_2MASS','H_MSIG_2MASS','K_MSIG_2MASS', 'W1SIGMPRO','W2SIGMPRO','W3SIGMPRO','W4SIGMPRO']
  # 1-sigma error on magnitudes of other survey bandpasses in VEGA magnitudes

  # These are the column names for the other magnitudes fluxes in the output flux catalog
  # These names have to be in the corresponding order to the vega_mag_names above
  vega_bandpass_names = [ '2MASS_j',\
			   '2MASS_h',\
			   '2MASS_ks',\
		           'WISE_w1',\
		           'WISE_w2',\
		           'WISE_w3',\
		           'WISE_w4']

  #-----------------------------------------------------------------------------
  # Convert all "empty" columns to np.NaNs
  #-----------------------------------------------------------------------------

  for i in range(len(sdss_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = quasar_catalog[sdss_mag_err_names[i]].replace(0.0,np.NaN)

      flux_catalog[sdss_bandpass_names[i]] = quasar_catalog[sdss_mag_names[i]].replace(0.0,np.NaN)

  for i in range(len(vega_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]] = quasar_catalog[vega_mag_err_names[i]].replace(0.0,np.NaN)

      flux_catalog[vega_bandpass_names[i]] = quasar_catalog[vega_mag_names[i]].replace(0.0,np.NaN)

  print flux_catalog.columns


  #-----------------------------------------------------------------------------
  # Convert SDSS magnitudes using the correct AB magnitude zero point and
  # deredden them for the flux catalog
  #-----------------------------------------------------------------------------

  # Conversion from nanomaggies to Jansky and correction to correct zero point flux
  for i in range(len(sdss_bandpass_names)):
      name = sdss_bandpass_names[i]

      # convert VEGA to AB magnitudes
      flux_catalog[name] = phot.VEGAtoAB(flux_catalog[name],name)
      # apply correct dereddening
      flux_catalog[name] = phot.deredden_mag(flux_catalog[name],name,quasar_catalog.EXTINCTION_U,'SDSS_u')

      flux_catalog[name] = phot.ASINHMAGtoFLUX(flux_catalog[name],name)



  #-----------------------------------------------------------------------------
  # Conversion of the Vega magnitudes from the other survey bandpasses to
  # fluxes, AB correctiond and reddening is applied before
  #-----------------------------------------------------------------------------

  for i in range(len(vega_bandpass_names)):

      # convert VEGA to AB magnitudes
      flux_catalog[vega_bandpass_names[i]] = phot.VEGAtoAB(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i])

      # apply the correct dereddening
      flux_catalog[vega_bandpass_names[i]] = phot.deredden_mag(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i],quasar_catalog.EXTINCTION_U,'SDSS_u')

      # convert exinction corrected AB magnitudes to fluxes in Jy
      flux_catalog[vega_bandpass_names[i]] = phot.ABMAGtoFLUX(flux_catalog[vega_bandpass_names[i]])

  #-----------------------------------------------------------------------------
  # Conversion of SDSS magnitude errors into 1-sigma uncertainties
  #-----------------------------------------------------------------------------

  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes

  for i in range(len(sdss_bandpass_names)):


      # calculating the lower and upper magnitude levels
      low_mag = np.array(quasar_catalog[sdss_mag_names[i]] + flux_catalog['sigma_'+sdss_bandpass_names[i]])
      up_mag =  np.array(quasar_catalog[sdss_mag_names[i]] - flux_catalog['sigma_'+sdss_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,sdss_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,sdss_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,sdss_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_U),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,sdss_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_U),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ASINHMAGtoFLUX(low_mag,sdss_bandpass_names[i])
      up_flux = phot.ASINHMAGtoFLUX(up_mag,sdss_bandpass_names[i])

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[sdss_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[sdss_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+sdss_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)


  #-----------------------------------------------------------------------------
  # Conversion of VEGA magnitudes 1-sigma uncertainties to 1-sigma
  # uncertainties on the extinction corrected fluxes
  #-----------------------------------------------------------------------------


  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes
  for i in range(len(vega_bandpass_names)):

      # calculating the lower and upper magnitude levels
      low_mag = np.array(quasar_catalog[vega_mag_names[i]] + flux_catalog['sigma_'+vega_bandpass_names[i]])
      up_mag =  np.array(quasar_catalog[vega_mag_names[i]] - flux_catalog['sigma_'+vega_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,vega_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,vega_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,vega_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_U),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,vega_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_U),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ABMAGtoFLUX(low_mag)
      up_flux = phot.ABMAGtoFLUX(up_mag)

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+vega_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)


  print flux_catalog.columns

  flux_catalog.to_csv('models/DR7_DR12Q_flux_cat.csv',index=False)





def build_flux_model_catalog_from_SDSS_star_cat(catalog_filename):

  try:
    quasar_catalog = pd.read_csv(catalog_filename)
  except:
    print "ERROR: Star catalog could not be read in. \n" \
      + "The star catalog has to be in CSV format."

  #-----------------------------------------------------------------------------
  # Building the flux catalog DataFrame from the Star catalog
  #-----------------------------------------------------------------------------

  #specifiec columns will be copied and renames, additional columns may be added by the user here
  general_column_names = ['RA','DEC','SDSSCLASS','CLASS','CLASSN', \
      'PSFMAG_U','PSFMAG_G','PSFMAG_R','PSFMAG_I','PSFMAG_Z', \
         'J_M_2MASS','H_M_2MASS','K_M_2MASS','W1MPRO','W2MPRO','W3MPRO','W4MPRO',\
             'PSFMAGERR_U','PSFMAGERR_G','PSFMAGERR_R','PSFMAGERR_I','PSFMAGERR_Z', \
         'J_MSIG_2MASS','H_MSIG_2MASS','K_MSIG_2MASS', 'W1SIGMPRO','W2SIGMPRO','W3SIGMPRO','W4SIGMPRO']

  flux_catalog = quasar_catalog[general_column_names].copy()

  flux_catalog.rename(columns={"RA": "ra","DEC":"dec","SDSSCLASS":"class_sdss","CLASS":"star_class","CLASSN":"class_num"},inplace=True)


  #-----------------------------------------------------------------------------
  #  Specify magnitudes from the quasar catalog to save in the new
  #  flux catalog
  #-----------------------------------------------------------------------------

  sdss_mag_names = ['PSFMAG_U','PSFMAG_G','PSFMAG_R','PSFMAG_I','PSFMAG_Z']
  # magnitudes in the SDSS bands u,g,r,i,z in asinh mags
  sdss_mag_err_names = ['PSFMAGERR_U','PSFMAGERR_G','PSFMAGERR_R','PSFMAGERR_I','PSFMAGERR_Z']
  # errors on the magnitudes in the u,g,r,i,z, bands in asinh mags

  #these are the column names for the SDSS fluxes in the output flux catalog
  sdss_bandpass_names =  ['SDSS_u',\
		  'SDSS_g',\
		  'SDSS_r',\
		  'SDSS_i',\
		  'SDSS_z']


  extinction_name = ['EXTINCTION_U']
  # Galactic extinction values in magnitudes for the SDSS bandpasses



  vega_mag_names =  ['J_M_2MASS','H_M_2MASS','K_M_2MASS','W1MPRO','W2MPRO','W3MPRO','W4MPRO']
  # magnitudes of other survey bandpasses in VEGA magnitudes
  vega_mag_err_names = ['J_MSIG_2MASS','H_MSIG_2MASS','K_MSIG_2MASS', 'W1SIGMPRO','W2SIGMPRO','W3SIGMPRO','W4SIGMPRO']
  # 1-sigma error on magnitudes of other survey bandpasses in VEGA magnitudes

  # These are the column names for the other magnitudes fluxes in the output flux catalog
  # These names have to be in the corresponding order to the vega_mag_names above
  vega_bandpass_names = [ '2MASS_j',\
			   '2MASS_h',\
			   '2MASS_ks',\
		           'WISE_w1',\
		           'WISE_w2',\
		           'WISE_w3',\
		           'WISE_w4']

  #-----------------------------------------------------------------------------
  # Convert all "empty" columns to np.NaNs
  #-----------------------------------------------------------------------------

  for i in range(len(sdss_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = quasar_catalog[sdss_mag_err_names[i]].replace(0.0,np.NaN)

      flux_catalog[sdss_bandpass_names[i]] = quasar_catalog[sdss_mag_names[i]].replace(0.0,np.NaN)

  for i in range(len(vega_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]] = quasar_catalog[vega_mag_err_names[i]].replace(0.0,np.NaN)

      flux_catalog[vega_bandpass_names[i]] = quasar_catalog[vega_mag_names[i]].replace(0.0,np.NaN)

  print flux_catalog.columns


  #-----------------------------------------------------------------------------
  # Convert SDSS magnitudes using the correct AB magnitude zero point and
  # deredden them for the flux catalog
  #-----------------------------------------------------------------------------

  # Conversion from nanomaggies to Jansky and correction to correct zero point flux
  for i in range(len(sdss_bandpass_names)):
      name = sdss_bandpass_names[i]

      # convert VEGA to AB magnitudes
      flux_catalog[name] = phot.VEGAtoAB(flux_catalog[name],name)
      # apply correct dereddening
      flux_catalog[name] = phot.deredden_mag(flux_catalog[name],name,quasar_catalog.EXTINCTION_U,'SDSS_u')

      flux_catalog[name] = phot.ASINHMAGtoFLUX(flux_catalog[name],name)



  #-----------------------------------------------------------------------------
  # Conversion of the Vega magnitudes from the other survey bandpasses to
  # fluxes, AB correctiond and reddening is applied before
  #-----------------------------------------------------------------------------

  for i in range(len(vega_bandpass_names)):

      # convert VEGA to AB magnitudes
      flux_catalog[vega_bandpass_names[i]] = phot.VEGAtoAB(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i])

      # apply the correct dereddening
      flux_catalog[vega_bandpass_names[i]] = phot.deredden_mag(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i],quasar_catalog.EXTINCTION_U,'SDSS_u')

      # convert exinction corrected AB magnitudes to fluxes in Jy
      flux_catalog[vega_bandpass_names[i]] = phot.ABMAGtoFLUX(flux_catalog[vega_bandpass_names[i]])

  #-----------------------------------------------------------------------------
  # Conversion of SDSS magnitude errors into 1-sigma uncertainties
  #-----------------------------------------------------------------------------

  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes

  for i in range(len(sdss_bandpass_names)):


      # calculating the lower and upper magnitude levels
      low_mag = np.array(quasar_catalog[sdss_mag_names[i]] + flux_catalog['sigma_'+sdss_bandpass_names[i]])
      up_mag =  np.array(quasar_catalog[sdss_mag_names[i]] - flux_catalog['sigma_'+sdss_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,sdss_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,sdss_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,sdss_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_U),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,sdss_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_U),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ASINHMAGtoFLUX(low_mag,sdss_bandpass_names[i])
      up_flux = phot.ASINHMAGtoFLUX(up_mag,sdss_bandpass_names[i])

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[sdss_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[sdss_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+sdss_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)


  #-----------------------------------------------------------------------------
  # Conversion of VEGA magnitudes 1-sigma uncertainties to 1-sigma
  # uncertainties on the extinction corrected fluxes
  #-----------------------------------------------------------------------------


  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes
  for i in range(len(vega_bandpass_names)):

      # calculating the lower and upper magnitude levels
      low_mag = np.array(quasar_catalog[vega_mag_names[i]] + flux_catalog['sigma_'+vega_bandpass_names[i]])
      up_mag =  np.array(quasar_catalog[vega_mag_names[i]] - flux_catalog['sigma_'+vega_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,vega_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,vega_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,vega_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_U),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,vega_bandpass_names[i],np.array(quasar_catalog.EXTINCTION_U),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ABMAGtoFLUX(low_mag)
      up_flux = phot.ABMAGtoFLUX(up_mag)

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+vega_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)


  print flux_catalog.columns

  flux_catalog.to_csv('models/DR10_star_flux_cat.csv',index=False)





def build_flux_model_catalog_from_my_allsky_cat(catalog_filename):

  try:
    catalog = pd.read_csv(catalog_filename)
  except:
    print "ERROR: Catalog could not be read in. \n" \
      + "The catalog has to be in CSV format."

  #-----------------------------------------------------------------------------
  # Building the flux catalog DataFrame from the catalog
  #-----------------------------------------------------------------------------

  #specifiec columns will be copied and renames, additional columns may be added by the user here
  general_column_names = ['sdss_ra','sdss_dec','wise_designation','specclass','specclass_person','photoz_mm','photoz_mms','photoz_rp',\
      'milliquas_ref_name','milliquas_ref_redshift','ned_flag','other_name',\
      'obs_class1','obs_class2','obs_z',\
     'psfmag_u','psfmag_g','psfmag_r','psfmag_i_1','psfmag_z', \
        'j_m_2mass','h_m_2mass','k_m_2mass','w1mpro','w2mpro','w3mpro','w4mpro',\
            'psfmagerr_u','psfmagerr_g','psfmagerr_r','psfmagerr_i','psfmagerr_z',\
         'j_msig_2mass','h_msig_2mass','k_msig_2mass', 'w1sigmpro','w2sigmpro','w3sigmpro','w4sigmpro',\
             'w1_unwise','w2_unwise']

  flux_catalog = catalog[general_column_names].copy()




  #-----------------------------------------------------------------------------
  #  Specify magnitudes from the quasar catalog to save in the new
  #  flux catalog
  #-----------------------------------------------------------------------------

  sdss_mag_names = ['psfmag_u','psfmag_g','psfmag_r','psfmag_i_1','psfmag_z']
  # magnitudes in the SDSS bands u,g,r,i,z in asinh mags
  sdss_mag_err_names = ['psfmagerr_u','psfmagerr_g','psfmagerr_r','psfmagerr_i','psfmagerr_z']
  # errors on the magnitudes in the u,g,r,i,z, bands in asinh mags

  #these are the column names for the SDSS fluxes in the output flux catalog
  sdss_bandpass_names =  ['SDSS_u',\
		  'SDSS_g',\
		  'SDSS_r',\
		  'SDSS_i',\
		  'SDSS_z']


  extinction_name = ['extinction_u']
  # Galactic extinction values in magnitudes for the SDSS bandpasses



  vega_mag_names =  ['j_m_2mass','h_m_2mass','k_m_2mass','w1mpro','w2mpro','w3mpro','w4mpro']
  # magnitudes of other survey bandpasses in VEGA magnitudes
  vega_mag_err_names = ['j_msig_2mass','h_msig_2mass','k_msig_2mass', 'w1sigmpro','w2sigmpro','w3sigmpro','w4sigmpro']
  # 1-sigma error on magnitudes of other survey bandpasses in VEGA magnitudes

  # These are the column names for the other magnitudes fluxes in the output flux catalog
  # These names have to be in the corresponding order to the vega_mag_names above
  vega_bandpass_names = [ '2MASS_j',\
			   '2MASS_h',\
			   '2MASS_ks',\
		           'WISE_w1',\
		           'WISE_w2',\
		           'WISE_w3',\
		           'WISE_w4']


  #UNWISE FLUX NAMES

  unwise_mag_names = ['w1_unwise','w2_unwise']
  unwise_bandpass_names = ['UNWISE_w1','UNWISE_w2']
  wise_equivalent_names = ['WISE_w1', 'WISE_w2']

  #-----------------------------------------------------------------------------
  # Convert all "empty" columns to np.NaNs
  #-----------------------------------------------------------------------------

  for i in range(len(sdss_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = catalog[sdss_mag_err_names[i]].replace(0.0,np.NaN)

      flux_catalog[sdss_bandpass_names[i]] = catalog[sdss_mag_names[i]].replace(0.0,np.NaN)

  for i in range(len(vega_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]] = catalog[vega_mag_err_names[i]].replace(0.0,np.NaN)

      flux_catalog[vega_bandpass_names[i]] = catalog[vega_mag_names[i]].replace(0.0,np.NaN)

  for i in range(len(unwise_bandpass_names)):

      flux_catalog[unwise_bandpass_names[i]] = catalog[unwise_mag_names[i]].replace(0.0,np.NaN)


  print flux_catalog.columns


  #-----------------------------------------------------------------------------
  # Convert SDSS magnitudes using the correct AB magnitude zero point and
  # deredden them for the flux catalog
  #-----------------------------------------------------------------------------

  # Conversion from nanomaggies to Jansky and correction to correct zero point flux
  for i in range(len(sdss_bandpass_names)):
      name = sdss_bandpass_names[i]

      # convert VEGA to AB magnitudes
      flux_catalog[name] = phot.VEGAtoAB(flux_catalog[name],name)
      # apply correct dereddening
      flux_catalog[name] = phot.deredden_mag(flux_catalog[name],name,catalog.extinction_u,'SDSS_u')

      flux_catalog[name] = phot.ASINHMAGtoFLUX(flux_catalog[name],name)



  #-----------------------------------------------------------------------------
  # Conversion of the Vega magnitudes from the other survey bandpasses to
  # fluxes, AB correctiond and reddening is applied before
  #-----------------------------------------------------------------------------

  for i in range(len(vega_bandpass_names)):

      # convert VEGA to AB magnitudes
      flux_catalog[vega_bandpass_names[i]] = phot.VEGAtoAB(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i])

      # apply the correct dereddening
      flux_catalog[vega_bandpass_names[i]] = phot.deredden_mag(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i],catalog.extinction_u,'SDSS_u')

      # convert exinction corrected AB magnitudes to fluxes in Jy
      flux_catalog[vega_bandpass_names[i]] = phot.ABMAGtoFLUX(flux_catalog[vega_bandpass_names[i]])

  # UNWISE FLUXES

  for i in range(len(unwise_bandpass_names)):

      # convert VEGA to AB magnitudes
      flux_catalog[unwise_bandpass_names[i]] = phot.VEGAtoAB(flux_catalog[unwise_bandpass_names[i]],wise_equivalent_names[i])

      # apply the correct dereddening
      flux_catalog[unwise_bandpass_names[i]] = phot.deredden_mag(flux_catalog[unwise_bandpass_names[i]],wise_equivalent_names[i],catalog.extinction_u,'SDSS_u')

      # convert exinction corrected AB magnitudes to fluxes in Jy
      flux_catalog[unwise_bandpass_names[i]] = phot.ABMAGtoFLUX(flux_catalog[unwise_bandpass_names[i]])

  #-----------------------------------------------------------------------------
  # Conversion of SDSS magnitude errors into 1-sigma uncertainties
  #-----------------------------------------------------------------------------

  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes

  for i in range(len(sdss_bandpass_names)):


      # calculating the lower and upper magnitude levels
      low_mag = np.array(catalog[sdss_mag_names[i]] + flux_catalog['sigma_'+sdss_bandpass_names[i]])
      up_mag =  np.array(catalog[sdss_mag_names[i]] - flux_catalog['sigma_'+sdss_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,sdss_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,sdss_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,sdss_bandpass_names[i],np.array(catalog.extinction_u),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,sdss_bandpass_names[i],np.array(catalog.extinction_u),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ASINHMAGtoFLUX(low_mag,sdss_bandpass_names[i])
      up_flux = phot.ASINHMAGtoFLUX(up_mag,sdss_bandpass_names[i])

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[sdss_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[sdss_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+sdss_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)


  #-----------------------------------------------------------------------------
  # Conversion of VEGA magnitudes 1-sigma uncertainties to 1-sigma
  # uncertainties on the extinction corrected fluxes
  #-----------------------------------------------------------------------------


  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes
  for i in range(len(vega_bandpass_names)):

      # calculating the lower and upper magnitude levels
      low_mag = np.array(catalog[vega_mag_names[i]] + flux_catalog['sigma_'+vega_bandpass_names[i]])
      up_mag =  np.array(catalog[vega_mag_names[i]] - flux_catalog['sigma_'+vega_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,vega_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,vega_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,vega_bandpass_names[i],np.array(catalog.extinction_u),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,vega_bandpass_names[i],np.array(catalog.extinction_u),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ABMAGtoFLUX(low_mag)
      up_flux = phot.ABMAGtoFLUX(up_mag)

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+vega_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)



  print flux_catalog.columns

  flux_catalog.to_csv('models/allsky_selection_cat.csv',index=False)







def build_flux_model_catalog_from_bright_cat(catalog_filename):

  try:
    catalog = pd.read_hdf(catalog_filename,'data')
  except:
    print "ERROR: Catalog could not be read in. \n" \
      + "The catalog has to be in CSV format."

  #-----------------------------------------------------------------------------
  # Building the flux catalog DataFrame from the catalog
  #-----------------------------------------------------------------------------

  #specifiec columns will be copied and renames, additional columns may be added by the user here
  #general_column_names = ['sdss_ra','sdss_dec','wise_designation', \
     #'psfmag_u','psfmag_g','psfmag_r','psfmag_i','psfmag_z', \
        #'j_m_2mass','h_m_2mass','k_m_2mass','w1mpro','w2mpro','w3mpro','w4mpro',\
            #'psfmagerr_u','psfmagerr_g','psfmagerr_r','psfmagerr_i','psfmagerr_z',\
         #'j_msig_2mass','h_msig_2mass','k_msig_2mass', 'w1sigmpro','w2sigmpro','w3sigmpro','w4sigmpro',\
             #'w1_unwise','w2_unwise']

  #flux_catalog = catalog[general_column_names].copy()
  flux_catalog = catalog.copy(deep=True)




  #-----------------------------------------------------------------------------
  #  Specify magnitudes from the quasar catalog to save in the new
  #  flux catalog
  #-----------------------------------------------------------------------------

  sdss_mag_names = ['psfmag_u','psfmag_g','psfmag_r','psfmag_i','psfmag_z']
  # magnitudes in the SDSS bands u,g,r,i,z in asinh mags
  sdss_mag_err_names = ['psfmagerr_u','psfmagerr_g','psfmagerr_r','psfmagerr_i','psfmagerr_z']
  # errors on the magnitudes in the u,g,r,i,z, bands in asinh mags

  #these are the column names for the SDSS fluxes in the output flux catalog
  sdss_bandpass_names =  ['SDSS_u',\
		  'SDSS_g',\
		  'SDSS_r',\
		  'SDSS_i',\
		  'SDSS_z']


  extinction_name = ['extinction_u']
  # Galactic extinction values in magnitudes for the SDSS bandpasses



  vega_mag_names =  ['j_m_2mass','h_m_2mass','k_m_2mass','w1mpro','w2mpro','w3mpro','w4mpro']
  # magnitudes of other survey bandpasses in VEGA magnitudes
  vega_mag_err_names = ['j_msig_2mass','h_msig_2mass','k_msig_2mass', 'w1sigmpro','w2sigmpro','w3sigmpro','w4sigmpro']
  # 1-sigma error on magnitudes of other survey bandpasses in VEGA magnitudes

  # These are the column names for the other magnitudes fluxes in the output flux catalog
  # These names have to be in the corresponding order to the vega_mag_names above
  vega_bandpass_names = [ '2MASS_j',\
			   '2MASS_h',\
			   '2MASS_ks',\
		           'WISE_w1',\
		           'WISE_w2',\
		           'WISE_w3',\
		           'WISE_w4']


  ##UNWISE FLUX NAMES

  #unwise_mag_names = ['w1_unwise','w2_unwise']
  #unwise_bandpass_names = ['UNWISE_w1','UNWISE_w2']
  #wise_equivalent_names = ['WISE_w1', 'WISE_w2']

  #-----------------------------------------------------------------------------
  # Convert all "empty" columns to np.NaNs
  #-----------------------------------------------------------------------------

  for i in range(len(sdss_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = catalog[sdss_mag_err_names[i]].replace(0.0,np.NaN)

      flux_catalog[sdss_bandpass_names[i]] = catalog[sdss_mag_names[i]].replace(0.0,np.NaN)

  for i in range(len(vega_bandpass_names)):

      # replace values that are 0 with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]] = catalog[vega_mag_err_names[i]].replace(0.0,np.NaN)

      flux_catalog[vega_bandpass_names[i]] = catalog[vega_mag_names[i]].replace(0.0,np.NaN)

  #for i in range(len(unwise_bandpass_names)):

      #flux_catalog[unwise_bandpass_names[i]] = catalog[unwise_mag_names[i]].replace(0.0,np.NaN)


  print flux_catalog.columns


  #-----------------------------------------------------------------------------
  # Convert SDSS magnitudes using the correct AB magnitude zero point and
  # deredden them for the flux catalog
  #-----------------------------------------------------------------------------

  # Conversion from nanomaggies to Jansky and correction to correct zero point flux
  for i in range(len(sdss_bandpass_names)):
      name = sdss_bandpass_names[i]

      # convert VEGA to AB magnitudes
      flux_catalog[name] = phot.VEGAtoAB(flux_catalog[name],name)
      # apply correct dereddening
      flux_catalog[name] = phot.deredden_mag(flux_catalog[name],name,catalog.extinction_u,'SDSS_u')

      flux_catalog[name] = phot.ASINHMAGtoFLUX(flux_catalog[name],name)



  #-----------------------------------------------------------------------------
  # Conversion of the Vega magnitudes from the other survey bandpasses to
  # fluxes, AB correctiond and reddening is applied before
  #-----------------------------------------------------------------------------

  for i in range(len(vega_bandpass_names)):

      # convert VEGA to AB magnitudes
      flux_catalog[vega_bandpass_names[i]] = phot.VEGAtoAB(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i])

      # apply the correct dereddening
      flux_catalog[vega_bandpass_names[i]] = phot.deredden_mag(flux_catalog[vega_bandpass_names[i]],vega_bandpass_names[i],catalog.extinction_u,'SDSS_u')

      # convert exinction corrected AB magnitudes to fluxes in Jy
      flux_catalog[vega_bandpass_names[i]] = phot.ABMAGtoFLUX(flux_catalog[vega_bandpass_names[i]])

  # UNWISE FLUXES

  #for i in range(len(unwise_bandpass_names)):

      ## convert VEGA to AB magnitudes
      #flux_catalog[unwise_bandpass_names[i]] = phot.VEGAtoAB(flux_catalog[unwise_bandpass_names[i]],wise_equivalent_names[i])

      ## apply the correct dereddening
      #flux_catalog[unwise_bandpass_names[i]] = phot.deredden_mag(flux_catalog[unwise_bandpass_names[i]],wise_equivalent_names[i],catalog.extinction_u,'SDSS_u')

      ## convert exinction corrected AB magnitudes to fluxes in Jy
      #flux_catalog[unwise_bandpass_names[i]] = phot.ABMAGtoFLUX(flux_catalog[unwise_bandpass_names[i]])

  #-----------------------------------------------------------------------------
  # Conversion of SDSS magnitude errors into 1-sigma uncertainties
  #-----------------------------------------------------------------------------

  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes

  for i in range(len(sdss_bandpass_names)):


      # calculating the lower and upper magnitude levels
      low_mag = np.array(catalog[sdss_mag_names[i]] + flux_catalog['sigma_'+sdss_bandpass_names[i]])
      up_mag =  np.array(catalog[sdss_mag_names[i]] - flux_catalog['sigma_'+sdss_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,sdss_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,sdss_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,sdss_bandpass_names[i],np.array(catalog.extinction_u),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,sdss_bandpass_names[i],np.array(catalog.extinction_u),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ASINHMAGtoFLUX(low_mag,sdss_bandpass_names[i])
      up_flux = phot.ASINHMAGtoFLUX(up_mag,sdss_bandpass_names[i])

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[sdss_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[sdss_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+sdss_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+sdss_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)


  #-----------------------------------------------------------------------------
  # Conversion of VEGA magnitudes 1-sigma uncertainties to 1-sigma
  # uncertainties on the extinction corrected fluxes
  #-----------------------------------------------------------------------------


  # We calculate the 1-sigma uncertainties on the fluxes by calculating the mean
  # of the difference between the upper and lower 1-sigma magnitude converted
  # fluxes
  for i in range(len(vega_bandpass_names)):

      # calculating the lower and upper magnitude levels
      low_mag = np.array(catalog[vega_mag_names[i]] + flux_catalog['sigma_'+vega_bandpass_names[i]])
      up_mag =  np.array(catalog[vega_mag_names[i]] - flux_catalog['sigma_'+vega_bandpass_names[i]])

      # convert VEGA to AB magnitudes
      low_mag = phot.VEGAtoAB(low_mag,vega_bandpass_names[i])
      up_mag = phot.VEGAtoAB(up_mag,vega_bandpass_names[i])

      # apply the correct dereddening
      low_mag = phot.deredden_mag(low_mag,vega_bandpass_names[i],np.array(catalog.extinction_u),'SDSS_u')
      up_mag = phot.deredden_mag(up_mag,vega_bandpass_names[i],np.array(catalog.extinction_u),'SDSS_u')

      # convert corrected AB magnitudes to fluxes in Jy
      low_flux = phot.ABMAGtoFLUX(low_mag)
      up_flux = phot.ABMAGtoFLUX(up_mag)

      # calculating the difference between the mean flux and the upper/lower values
      sigma_low = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-low_flux)
      sigma_up = np.abs(np.array(flux_catalog[vega_bandpass_names[i]])-up_flux)

      # calculating the mean deviation
      flux_catalog['sigma_'+vega_bandpass_names[i]] = 0.5*(sigma_low+sigma_up)

      # replacing infinities with np.NaN
      flux_catalog['sigma_'+vega_bandpass_names[i]].replace(np.inf,np.NaN,inplace=True)



  print flux_catalog.columns

  flux_catalog.to_hdf('models/bright_cat.hdf5','data')



build_flux_model_catalog_from_bright_cat('models/wise_tmass_sdss_bright.hdf5')

#build_flux_model_catalog_from_SDSS_QSO_cat("../DR12Q.csv")

#build_flux_model_catalog_from_joint_SDSS_QSO_cat('../catalogs/dr12dr7qso_nonbal_phot.csv')

#build_flux_model_catalog_from_SDSS_star_cat('../catalogs/dr10star_phot2.csv')

#build_flux_model_catalog_from_my_allsky_cat('../catalogs/wise_tmass_sdss_allsky_photoz_milliquas_other_ned_tmassfull_unwise_myobs_s16.csv')
