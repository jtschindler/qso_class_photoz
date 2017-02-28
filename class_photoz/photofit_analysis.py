import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import LogNorm

# pdf plot
# flux_ratio redshift relation plots for model catalog
# flux_ratio redshift relation with median and/or spline


def set_pred_classes(df_pred):

    # if qso redchisq < star redchisq set mult class to star class
    df_pred.loc[
        df_pred.query('pf_qso_redchisq < pf_star_redchisq').index ,
         'mult_class_pred'] = \
         df_pred.query('pf_qso_redchisq < pf_star_redchisq')['qso_class']

    # if qso redchisq >= star redchisq set mult class to qso class
    df_pred.loc[
        df_pred.query('pf_qso_redchisq >= pf_star_redchisq').index ,
         'mult_class_pred'] = \
         df_pred.query('pf_qso_redchisq >= pf_star_redchisq')['pf_star_class']

    # if qso redchisq < star redchisq set binary class to QSO
    df_pred.loc[df_pred.query('pf_qso_redchisq < pf_star_redchisq').index ,
     'bin_class_pred'] = 'QSO'

    # if qso redchisq >= star redchisq set binary class to STAR
    df_pred.loc[df_pred.query('pf_qso_redchisq >= pf_star_redchisq').index ,
     'bin_class_pred'] = 'STAR'

    return df_pred

def set_redshift_classes(df, z_label, class_label):

    # Create new classification column and set all values to "null"
    df[class_label] = 'null'

    # lower and upper redshift boundaries for the classes
    lowz=[0,1.5,2.2,3.5]
    highz=[1.5,2.2,3.5,10]

    # names of the classes
    labels=['vlowz','lowz','midz','highz']

    # set classes according to boundaries and names above
    for idx in range(len(lowz)):
        df.loc[
                df.query(str(lowz[idx])+'<' + str(z_label) + '<='+str(highz[idx])).index,
                class_label] = labels[idx]

    return df


# # Plot color z relations
# def plot_fr_z_relations(model_catalog,flux_ratio_names,redshift_bins,bin_data):
#     """This function plots the flux-ratio redshift realtions for all flux_ratios
#     in the model_catalog.
#
#
#     Parameters:
#         model_catalog : dataframe
#         Dataframe with objects from which the flux-ratio model is calculated
#
#         flux_ratio_names : list of strings
#         Names of the flux ratios in the model catalog
#
#         redshift_bins :
#
#         bin_data : array-like ([ number of bins ]x4)
#         This array contains the information about the bins. It contains the
#         lower redshift boundary, the upper redshift boundary, the mean
#         redshift and the redshift width in this order.
#
#     Returns:
#         plt : matplotlib plot
#
#     """
#
#     for i in range(len(flux_ratio_names)):
#         fig = plt.figure(num=None,figsize=(8,5), dpi=100)
#         ax = fig.add_subplot(1,1,1)
#
#         ax.plot(model_catalog.redshift,model_catalog[flux_ratio_names[i]],'k.')
#         ax.plot(bin_data[2,:],redshift_bins.agg([np.mean])[flux_ratio_names[i]],'b-',linewidth=4)
#         ax.plot(bin_data[2,:],redshift_bins.agg([np.median])[flux_ratio_names[i]],'r-',linewidth=4)
#         ax.plot(bin_data[2,:],redshift_bins.agg(lambda x: np.percentile(x[flux_ratio_names[i]], q = 16))[flux_ratio_names[i]],'m-',linewidth=2)
#         ax.plot(bin_data[2,:],redshift_bins.agg(lambda x: np.percentile(x[flux_ratio_names[i]], q = 84))[flux_ratio_names[i]],'m-',linewidth=2)
#
#         ax.set_xlabel('redshift')
#         ax.set_ylabel(flux_ratio_names[i])
#         ax.set_ylim(-1,2)
#         ax.set_xlim(0,5)
#
#         plt.show()




def plot_star_classes(obj_catalog):
    """This function plots the identified stellar classification of each object
    in the obj_catalog against it's photometric stellar classification.

    Parameters:
        obj_catalog : dataframe
        This dataframe contains the photometric stellar classification for
        every source as well as it's original spectral classification.

    Returns:
        plt : matplotlib plot

    """

    fig = plt.figure(num=None,figsize=(8,8), dpi=100)
    ax = fig.add_subplot(1,1,1)

    phot_class = obj_catalog.phot_star_class
    sclass = obj_catalog.star_class
    phot_class_num = np.zeros(obj_catalog.shape[0])
    sclass_num = np.zeros(obj_catalog.shape[0])

    star_classes = ['WD',\
                  'O','O8','O9','OB','B0','B1','B2','B3','B5','B6','B7','B8','B9',\
                  'A0','A1','A2','A3','A4','A5','A6','A8','A9',\
                  'F0','F2','F3','F5','F6','F8','F9',\
                  'G0','G1','G2','G3','G4','G5','G8','G9',\
                  'K0','K1','K2','K3','K4','K5','K7',\
                'M0','M1','M2','M3','M4','M5','M6','M7','M8','M9', \
                    'L0','L1','L2','L3','L4','L5','L9','Ldwarf', \
                      'T','other','C']
    print len(star_classes)

    star_dict = dict(zip(star_classes,np.arange(len(star_classes))))

    # print phot_class.value_counts()

    for i in range(len(phot_class)):
        print phot_class[i], star_dict[phot_class[i]], sclass[i],star_dict[sclass[i]]
        phot_class_num[i] = star_dict[phot_class[i]]
        sclass_num[i] = star_dict[sclass[i]]

    #ax.plot(sclass_num,phot_class_num,'.')

    cmap = plt.cm.Blues
    cmap.set_bad('0.85',1.0)

    cax = plt.hist2d(sclass_num,phot_class_num, bins=65,range =  [[0,65], [0,65]], norm = LogNorm(), cmap=cmap, zorder=0)
    cbar = plt.colorbar(ticks=[1,5,10,15,20,25,30,40])
    cbar.ax.set_yticklabels([1,5,10,15,20,25,30,40],fontsize=12)

    ax.plot(np.arange(65),np.arange(65),'r')

    plt.xticks(np.arange(len(star_classes)),star_classes,fontsize=8,rotation='vertical')
    plt.yticks(np.arange(len(star_classes)),star_classes,fontsize=8)

    plt.grid(True)
    return plt




# def spline_plot(model_catalog,spline_list,flux_model,flux_ratio_names,work_dir,smoothing):
#
#   x = np.arange(0.0,5,0.1)
#
#   for i in range(len(flux_ratio_names)):
#       fig = plt.figure(num=None,figsize=(8,5), dpi=80)
#       ax = fig.add_subplot(1,1,1)
#
#       ax.plot(model_catalog.redshift,model_catalog[flux_ratio_names[i]],'k.')
#       ax.plot(flux_model.bin_middle,flux_model[flux_ratio_names[i]],'b',linewidth=2)
#       spl = spline_list[i]
#       spl.set_smoothing_factor(smoothing[i])
#       ax.plot(x,spl(x),'r',linewidth=2)
#
#       ax.set_xlabel('redshift')
#       ax.set_ylabel(flux_ratio_names[i])
#       ax.set_ylim(-1,2)
#       ax.set_xlim(0,5)
#
#       fname = work_dir+'spline_'+flux_ratio_names[i]+'_redshift.pdf'
#
#       plt.show()
#       #plt.savefig(fname, format='pdf')
#       plt.close(fig)


# def star_quasar_class_test_plot(obj_catalog,work_dir):
#
#     fig = plt.figure(num=None,figsize=(8,8), dpi=100)
#     ax = fig.add_subplot(1,1,1)
#
#     #obj_catalog = obj_catalog.query('photo_z > 2.5 and PSFMAG_I < 18.5')
#
#     qsos = obj_catalog.query('redshift > 0')
#
#     stars= obj_catalog.query('class_num > 0')
#
#
#     ax.scatter(qsos.phot_qso_redchisq,qsos.phot_star_redchisq,c='b', s=10, marker='D',zorder=4, edgecolors='b')
#     ax.scatter(stars.phot_qso_redchisq,stars.phot_star_redchisq,c='r', s=10, marker='D',zorder=4, edgecolors='r')
#
#     ax.text(0.2, 0.7,'Quasars: '+str(qsos.query('phot_star_redchisq > phot_qso_redchisq').shape[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#     ax.text(0.2, 0.75,'Stars: '+str(stars.query('phot_star_redchisq > phot_qso_redchisq').shape[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#
#     ax.text(0.5, 0.7,'Quasars: '+str(qsos.query('phot_star_redchisq < phot_qso_redchisq').shape[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#     ax.text(0.5, 0.75,'Stars: '+str(stars.query('phot_star_redchisq < phot_qso_redchisq').shape[0]), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
#
#     x = np.arange(0,50,0.1)
#
#     ax.plot(x,x,'k')
#
#     ax.set_xlim(0,20)
#     ax.set_ylim(0,10)
#
#     ax.set_xlabel(r'$\chi^2_{\rm{red}}$(QSO)')
#     ax.set_ylabel(r'$\chi^2_{\rm{red}}$(STAR)')
#
#     fname = work_dir+'chisq.pdf'
#
#     return plt
