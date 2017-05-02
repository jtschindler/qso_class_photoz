import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.colors import LogNorm

import sklearn.metrics as met
from class_photoz import photoz_analysis as pz_an
from class_photoz import ml_analysis as ml_an

# pdf plot
# flux_ratio redshift relation plots for model catalog
# flux_ratio redshift relation with median and/or spline

def photoz_analysis(df_pred, z_label_pred, z_label_true):

    df = df_pred.copy(deep=True)
    df = df.query('bin_class_true == "QSO"')

    z_true = df[z_label_true].values
    z_pred = df[z_label_pred].values

    print("The r2 score for the Photometric Redshift Estimation is:\t"),  met.r2_score(z_true, z_pred)

    pz_an.plot_redshifts(z_true,z_pred)
    pz_an.plot_error_hist(z_true,z_pred)

    plt.show()

def classification_analysis(y_true,y_pred,labels):

    print("Detailed classification report:")
    print()
    print("The model is trained on the training set.")
    print("The scores are computed on the test set.")
    print()
    y_true = y_true.astype('string')
    y_pred = y_pred.astype('string')

    print(met.classification_report(y_true, y_pred))
    print()

    cnf_matrix = met.confusion_matrix(y_true, y_pred, labels=labels, sample_weight=None)

    ml_an.plot_confusion_matrix(cnf_matrix, classes=labels, normalize=True,
                      title='Confusion matrix, with normalization')

    ml_an.plot_confusion_matrix(cnf_matrix, classes=labels, normalize=False,
                      title='Confusion matrix, without normalization')


    plt.show()


def set_pred_classes(df_pred):

    # if qso redchisq < star redchisq set mult class to star class
    idx =  df_pred.query('pf_qso_redchisq < pf_star_redchisq').index
    df_pred.loc[ idx, 'pf_mult_class_pred'] = df_pred.loc[idx,'pf_qso_class']

    # if qso redchisq >= star redchisq set mult class to qso class
    idx =  df_pred.query('pf_qso_redchisq >= pf_star_redchisq').index
    df_pred.loc[ idx, 'pf_mult_class_pred'] = df_pred.loc[idx,'pf_star_class']

    # if qso redchisq < star redchisq set binary class to QSO
    idx =  df_pred.query('pf_qso_redchisq < pf_star_redchisq').index
    df_pred.loc[idx, 'pf_bin_class_pred'] = 'QSO'

    # if qso redchisq >= star redchisq set binary class to STAR
    idx =  df_pred.query('pf_qso_redchisq >= pf_star_redchisq').index
    df_pred.loc[ idx, 'pf_bin_class_pred'] = 'STAR'

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
