import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import photoz_analysis as pz_an
import ml_analysis as ml_an


def exclude_known_obj(df,verbose=False):
    # -----------------------------------------------------------------------------
    # Exclude known quasars
    # -----------------------------------------------------------------------------


    # 1) exclude all known objects from SDSS
    df = df.query('specclass == "null"')

    if verbose :
        print "\n"
        print str(df.shape[0])+' point sources without SDSS automatic spectral classification'


    # 2) exclude all known objects matched with MILLIQUAS
    df['milliquas_ref_name'].replace(np.NaN,'None',inplace=True)

    # df = df.query('milliquas_ref_name == "None" or milliquas_ref_name == "NBCK 3  " or milliquas_ref_name == "XDQSO   " or milliquas_ref_name =="NBCKDE  "')
    df = df.query('milliquas_ref_name == "None" or milliquas_ref_name == "NBCK 3" or milliquas_ref_name == "XDQSO" or milliquas_ref_name =="NBCKDE"')
    #df = df.query('milliquas_ref_redshift != "MQ          "')
    df = df.query('milliquas_ref_redshift != "MQ"')

    if verbose :
        print str(df.shape[0])+' point sources exlcuding known QSOs in MILLIQUAS'


    # 3) exclude all known quasars  to NED
    df = df.query('ned_flag == False')

    if verbose :
        print str(df.shape[0])+' point sources exlcuding known QSOs in NED'


    # 4) exclude all other known QSOs

    df['other_name'].replace(np.NaN,'None',inplace=True)

    df = df.query('other_name == "None"')

    if verbose :
        print str(df.shape[0])+' point sources exlcuding known QSOs from other sources'
        print "\n"


    return df


def calc_priorities_old(cat):


    cat.loc[:,'priority_photoz'] = 0
    cat.loc[:,'priority_photoz_z'] = 0
    cat.loc[:,'priority_rf'] = 0
    cat.loc[:,'priority_milliquas'] = 0
    cat.loc[:,'priority_rf_z'] = 0
    cat.loc[:,'priority_milliquas_z'] = 0
    cat.loc[:,'priority'] = 0
    cat.loc[:,'priority_sum'] = 0
    cat.loc[:,'priority_z_sum'] = 0
    cat.loc[:,'priority_ref'] = None


    # i-band cut
    i_band_cut = ' and psfmag_i < 18.5'

    milliquas_a = 'milliquas_qso_prob >= 90'+i_band_cut
    milliquas_b = '3.5>milliquas_z >=2.8'
    milliquas_c = 'milliquas_z >=3.5'

    photoz_a = '(photoz_mms - photoz_mm >= 0) or ' + \
    '(photoz_mm < 1.0 and (photoz_type_star == "T2" or photoz_type_star == "L5.5"))' \
                +i_band_cut
    photoz_b = '3.5>photoz_rp >= 2.8'
    photoz_c = 'photoz_rp >= 3.5'


    cat.loc[cat.query(milliquas_a).index,'priority_milliquas'] += 1
    cat.loc[cat.query(milliquas_b).index,'priority_milliquas_z'] += 1
    cat.loc[cat.query(milliquas_c).index,'priority_milliquas_z'] += 2


    cat.loc[cat.query(photoz_a).index,'priority_photoz'] += 1
    cat.loc[cat.query(photoz_b).index,'priority_photoz_z'] += 1
    cat.loc[cat.query(photoz_c).index,'priority_photoz_z'] += 2

    print cat.priority_milliquas.value_counts()

    print cat.priority_photoz.value_counts()

    cat = cat.query('priority_photoz > 0 or priority_milliquas > 0')
    print str(cat.shape[0])+' point sources that pass either photoz or milliquas criteria'

    crit1 = 'priority_milliquas_z >= priority_photoz_z'
    crit2 = 'priority_milliquas_z < priority_photoz_z'

    cat.loc[cat.query(crit1).index,'priority_ref'] = 'milliquas'
    cat.loc[cat.query(crit1).index,'priority'] = cat.loc[cat.query(crit1).index,'priority_milliquas_z']

    cat.loc[cat.query(crit2).index,'priority_ref'] = 'photoz'
    cat.loc[cat.query(crit2).index,'priority'] = cat.loc[cat.query(crit2).index,'priority_photoz_z']

    cat.priority_sum = cat.priority_photoz_z + cat.priority_milliquas_z

    print "Priority counts of total sample:\n"
    print cat.priority_milliquas_z.value_counts()
    print cat.priority_photoz_z.value_counts()

    print cat.priority.value_counts()
    print cat.priority_sum.value_counts()
    print cat.priority_ref.value_counts()

    return cat

def calc_priorities(cat):

    cat.loc[:,'priority_photoz'] = 0
    cat.loc[:,'priority_photoz_z'] = 0
    cat.loc[:,'priority_rf'] = 0
    cat.loc[:,'priority_milliquas'] = 0
    cat.loc[:,'priority_rf_z'] = 0
    cat.loc[:,'priority_milliquas_z'] = 0
    cat.loc[:,'priority'] = 0
    cat.loc[:,'priority_sum'] = 0
    cat.loc[:,'priority_z_sum'] = 0
    cat.loc[:,'priority_ref'] = None

    # i-band cut
    i_band_cut = ' and psfmag_i < 18.5'

    milliquas_a = 'milliquas_qso_prob >= 90'+i_band_cut
    milliquas_b = '3.5>milliquas_z >=2.8'
    milliquas_c = 'milliquas_z >=3.5'

    rf_a = 'rf_qso_prob > 0.5'+i_band_cut
    rf_b = '3.5> rf_photoz >= 2.8'
    rf_c = 'rf_photoz >= 3.5'

    photoz_a = '(photoz_mms - photoz_mm >= 0) or ' + \
    '(photoz_mm < 1.0 and (photoz_type_star == "T2" or photoz_type_star == "L5.5"))' \
                +i_band_cut
    photoz_b = '3.5>photoz_rp >= 2.8'
    photoz_c = 'photoz_rp >= 3.5'

    cat.loc[cat.query(milliquas_a).index,'priority_milliquas'] = 1
    cat.loc[cat.query(milliquas_b).index,'priority_milliquas_z'] = 1
    cat.loc[cat.query(milliquas_c).index,'priority_milliquas_z'] = 2

    cat.loc[cat.query(photoz_a).index,'priority_photoz'] = 1
    cat.loc[cat.query(photoz_b).index,'priority_photoz_z'] = 1
    cat.loc[cat.query(photoz_c).index,'priority_photoz_z'] = 2

    cat.loc[cat.query(rf_a).index,'priority_rf'] = 1
    cat.loc[cat.query(rf_b).index,'priority_rf_z'] = 1
    cat.loc[cat.query(rf_c).index,'priority_rf_z'] = 2

    print "Selection properties of Milliquas criteria (1 = selected)"
    print cat.priority_milliquas.value_counts()
    print "\n"
    print "Selection properties of RF criteria (1 = selected)"
    print cat.priority_rf.value_counts()
    print "\n"
    print "Selection properties of Color fitting criteria (1 = selected)"
    print cat.priority_photoz.value_counts()
    print "\n"

    crit1 = 'priority_milliquas_z >= priority_photoz_z and priority_milliquas_z >= priority_rf_z'+ \
            ' and (priority_rf > 0 or priority_milliquas > 0 or priority_photoz > 0)'
    crit2 = 'priority_rf_z > priority_milliquas_z and priority_rf_z >= priority_photoz_z'+ \
            ' and (priority_rf > 0 or priority_milliquas > 0 or priority_photoz > 0)'
    crit3 = 'priority_photoz_z > priority_milliquas_z and priority_photoz_z > priority_rf_z'+ \
            ' and (priority_rf > 0 or priority_milliquas > 0 or priority_photoz > 0)'

    cat.loc[cat.query(crit1).index,'priority_ref'] = 'milliquas'
    cat.loc[cat.query(crit1).index,'priority'] = \
                cat.loc[cat.query(crit1).index,'priority_milliquas_z']

    cat.loc[cat.query(crit2).index,'priority_ref'] = 'rf'
    cat.loc[cat.query(crit2).index,'priority'] = \
                cat.loc[cat.query(crit2).index,'priority_rf_z']

    cat.loc[cat.query(crit3).index,'priority_ref'] = 'photoz'
    cat.loc[cat.query(crit3).index,'priority'] = \
                cat.loc[cat.query(crit3).index,'priority_photoz_z']

    cat.loc[:,'priority_sum'] = cat.priority_rf + cat.priority_milliquas \
                        + cat.priority_photoz

    cat.loc[:,'priority_z_sum'] = cat.priority_rf_z + cat.priority_milliquas_z \
                        + cat.priority_photoz_z


    print "Priority counts of total sample:\n"
    print "Milliquas redshift priority :"
    print cat.priority_milliquas_z.value_counts()
    print "\n"
    print "Random Forest redshift priority :"
    print cat.priority_rf_z.value_counts()
    print "\n"
    print "Color fitting redshift priority :"
    print cat.priority_photoz_z.value_counts()
    print "\n"

    print "Total priority :"
    print cat.priority.value_counts()
    print "\n"
    print "Sum of selection priorities "
    print cat.priority_sum.value_counts()
    print "\n"
    print "Sum of redshift priorities "
    print cat.priority_z_sum.value_counts()
    print "\n"

    return cat


def select_qso_candidates(cat):

    # Exclude previously known objects
    cat = exclude_known_obj(cat)
    # Exclude already observed objects

    cat['obs_class1'].replace(np.NaN,'None',inplace=True)
    cat.query('obs_class1 == "None"',inplace=True)

    cat.query('psfmag_i < 18.5',inplace=True)
    cat = cat.query('priority_rf > 0 or priority_milliquas > 0 or priority_photoz > 0')
    #-----------------------------------------------------------------------
    #----------------------select observable sky----------------------------
    #-----------------------------------------------------------------------

    # SPRING
    spring = cat.query('90 <= sdss_ra < 270 and -10 <= sdss_dec <50')

    print "Priority counts of spring sample:\n"
    print spring.priority_milliquas_z.value_counts()
    print spring.priority_rf_z.value_counts()
    print spring.priority_photoz_z.value_counts()
    print spring.priority.value_counts()
    print spring.priority_sum.value_counts()
    print spring.priority_z_sum.value_counts()
    print spring.priority_ref.value_counts()
    print "\n"

    # FALL
    fall = cat.query('(sdss_ra >= 270 or sdss_ra <90) and -10 <= sdss_dec <50')

    print "Priority counts of fall sample:\n"
    print fall.priority_milliquas_z.value_counts()
    print fall.priority_rf_z.value_counts()
    print fall.priority_photoz_z.value_counts()
    print fall.priority.value_counts()
    print fall.priority_sum.value_counts()
    print fall.priority_z_sum.value_counts()
    print fall.priority_ref.value_counts()
    print "\n"



def select_known_qsos(df):

    df['milliquas_ref_name'].replace(np.NaN,'None',inplace=True)
    df['other_name'].replace(np.NaN,'None',inplace=True)

    sdss_crit = 'specclass == "QSO"'
    milliquas_crit = 'milliquas_ref_redshift == "MQ" or (milliquas_ref_name != "None" and milliquas_ref_name != "NBCK 3" and milliquas_ref_name != "XDQSO" and milliquas_ref_name !="NBCKDE")'
    ned_crit = 'ned_flag == True'
    other_crit = 'other_name != "None"'

    q = df.query(sdss_crit+' or '+milliquas_crit+' or '+ned_crit+' or '+other_crit)

    # Select known redshifts
    q.loc[:,'known_z'] = np.NaN


    q.loc[q.query(milliquas_crit).index,'known_z'] = q.loc[q.query(milliquas_crit).index,'milliquas_z']
    q.loc[q.query(sdss_crit).index,'known_z'] = q.loc[q.query(sdss_crit).index,'z']
    q.loc[q.query(ned_crit).index,'known_z'] = q.loc[q.query(ned_crit).index,'ned_redshift']
    q.loc[q.query(other_crit).index,'known_z'] = q.loc[q.query(other_crit).index,'other_redshift']

    return q


def eval_quasar_star_pred(df):
    print "\n"
    print "-------------------------------------------------------------------"
    print " Evaluate QSO/STAR classification of before known objects"
    print "-------------------------------------------------------------------"
    print "\n"

    print str(df.shape[0])+' point sources in full allsky catalog'

    known_qsos = select_known_qsos(df)

    print str(known_qsos.shape[0])+' known QSOs in full allsky catalog'
    print "\n"
    # Fraction of predicted QSO by rf to known QSOs

    pred_qsos = known_qsos.query('rf_qso_prob > 0.5')
    print "Fraction of predicted QSOs by RF to known QSOs in allsky catalog"
    print 1.0*pred_qsos.shape[0]/known_qsos.shape[0]
    print "\n"

    # Fraction of predicted QSO by color-z fitting to known QSOs

    pred_qsos = known_qsos.query('(photoz_mms >= photoz_mm) or ' + \
    '(photoz_mm < 1.0 and (photoz_type_star == "T2" or photoz_type_star == "L5.5"))')
    print "Fraction of predicted QSOs by color fitting to known QSOs in allsky catalog"
    print 1.0*pred_qsos.shape[0]/known_qsos.shape[0]
    print "\n"


    known_qsos = known_qsos.query('z > 2.8 or milliquas_z > 2.8 ')
    print str(known_qsos.shape[0])+' known z>2.8 SDSS+Milliquas QSOs in full allsky catalog'
    # Fraction of predicted QSO by rf to known QSOs

    pred_qsos = known_qsos.query('rf_qso_prob > 0.5')
    print "Fraction of predicted QSOs by RF to known QSOs in allsky catalog"
    print 1.0*pred_qsos.shape[0]/known_qsos.shape[0]
    print "\n"

    # Fraction of predicted QSO by color-z fitting to known QSOs

    pred_qsos = known_qsos.query('(photoz_mms >= photoz_mm) or ' + \
    '(photoz_mm < 1.0 and (photoz_type_star == "T2" or photoz_type_star == "L5.5"))')
    print "Fraction of predicted QSOs by color fitting to known QSOs in allsky catalog"
    print 1.0*pred_qsos.shape[0]/known_qsos.shape[0]
    print "\n"



    known_stars = df.query('specclass == "STAR"')
    print "\n"
    print str(known_stars.shape[0])+' known SDSS STARs in full allsky catalog'

    # Fraction of predicted QSO by rf to known STARs

    pred_qsos = known_stars.query('rf_qso_prob > 0.5')
    print "Fraction of predicted QSOs by RF to known STARs in allsky catalog"
    print 1.0*pred_qsos.shape[0]/known_stars.shape[0]
    print "\n"
    # Fraction of predicted QSO by color-z fitting to known STARs

    pred_qsos = known_stars.query('(photoz_mms >= photoz_mm) or ' + \
    '(photoz_mm < 1.0 and (photoz_type_star == "T2" or photoz_type_star == "L5.5"))')
    print "Fraction of predicted QSOs by color fitting to known STARs in allsky catalog"
    print 1.0*pred_qsos.shape[0]/known_stars.shape[0]
    print "\n"


def eval_photoz_pred(df):
    print "\n"
    print "-------------------------------------------------------------------"
    print " Evaluate photometric redshift prediction of before known objects"
    print "-------------------------------------------------------------------"
    print "\n"
    known_qsos = select_known_qsos(df)
    known_qsos['known_z'].replace(np.NaN,-99,inplace=True)
    known_qsos.query('known_z != -99',inplace=True)


    print"Evaluate photometric redshift of Random Forest method:"
    pz_an.evaluate_photoz(known_qsos.known_z.values,known_qsos.rf_photoz.values)
    pz_an.plot_redshifts(known_qsos.known_z.values,known_qsos.rf_photoz.values)
    plt.show()
    print "\n"

    print "Evaluate photometric redshift of Color-z fitting"
    pz_an.evaluate_photoz(known_qsos.known_z.values,known_qsos.photoz_rp.values)
    pz_an.plot_redshifts(known_qsos.known_z.values,known_qsos.photoz_rp.values)
    plt.show()
    print "\n"

    # Evaluate photometric redshift method against another
    pz_an.plot_redshifts(known_qsos.rf_photoz.values,known_qsos.photoz_rp.values)
    plt.show()




def eval_observed_cand(cat):
    print "\n"
    print "-------------------------------------------------------------------"
    print " Evaluate observed objects against selection metods"
    print "-------------------------------------------------------------------"
    # Testing the quasar/star selection
    cat.query('psfmag_i < 18.5',inplace=True)

    print "Number of total objects in i-band magnitude cut (psfmag_i < 18.5) sample: "
    print cat.obs_class1.value_counts()
    print "\n"
    mq_sel = cat.query('priority_milliquas > 0')

    print "Classes of observed objects that would have been selected by Milliquas"
    print mq_sel.obs_class1.value_counts()
    # print mq_sel[['obs_class1','priority_milliquas_z','obs_z']].query('obs_class1 =="QSO"')
    print "\n"

    rf_sel = cat.query('priority_rf > 0')

    print "Classes of observed objects that would have been selected by Random Forest"
    print rf_sel.obs_class1.value_counts()
    # print rf_sel[['obs_class1','priority_rf_z','obs_z']].query('obs_class1 =="QSO"')
    print "\n"

    pz_sel = cat.query('priority_photoz > 0')

    print "Classes of observed objects that would have been selected by Color fitting"
    print pz_sel.obs_class1.value_counts()
    # print pz_sel[['obs_class1','priority_photoz_z','obs_z']].query('obs_class1 =="QSO"')
    print "\n"

    print "Number of objects that would have been selected by 2 or more methods"
    print cat.query('priority_sum >= 2').obs_class1.value_counts()
    print "Number of objects that would have been selected by all three methods"
    print cat.query('priority_sum >= 3').obs_class1.value_counts()
    print "\n"

    # Investigate non selected but observed QSOs
    # print cat[['obs_class1','priority','obs_z','obs_class2','obs_date','psfmag_i']].query('priority == 0 and obs_class1=="QSO"')
    # -> It seems that these were selected using the previous selection criteria relying on color cuts
    # -> Or that these objects are low redshift objects that should not have been selected
    # -> Nearly half of them seem to be BALs
    # -> RF has highest completeness, Color-z has higest efficiency


    # Investigate photometric redshift outliers
    qsos = cat.query('obs_class1 == "QSO" and obs_z >= 0')
    print "Redshift outliers (delta z > 0.3) between observed objects and RF prediction"
    print qsos[['obs_class1','obs_class2','priority','obs_z','rf_photoz','photoz_rp','milliquas_z']].query('abs(obs_z - rf_photoz) > 0.3')
    num = qsos[['obs_class1','obs_class2','priority','obs_z','rf_photoz','photoz_rp','milliquas_z']].query('abs(obs_z - rf_photoz) > 0.3').shape[0]
    print "RF photometric redshift outliers are about" ,100*num/qsos.query('rf_photoz >= 0.0').shape[0], "%"

    print "\n"
    print "Redshift outliers (delta z > 0.3) between observed objects and Color-z prediction"
    print qsos[['obs_class1','obs_class2','priority','obs_z','rf_photoz','photoz_rp','milliquas_z']].query('abs(obs_z - photoz_rp) > 0.3')
    num = qsos[['obs_class1','obs_class2','priority','obs_z','rf_photoz','photoz_rp','milliquas_z']].query('abs(obs_z - photoz_rp) > 0.3').shape[0]
    print "Color-z photometric redshift outliers are about" ,100*num/qsos.query('photoz_rp >= 0.0').shape[0], "%"

    print "\n"
    print "Redshift outliers (delta z > 0.3) between observed objects and RF prediction"
    print qsos[['obs_class1','obs_class2','priority','obs_z','rf_photoz','photoz_rp','milliquas_z']].query('abs(obs_z - milliquas_z) > 0.3')
    num = qsos[['obs_class1','obs_class2','priority','obs_z','rf_photoz','photoz_rp','milliquas_z']].query('abs(obs_z - milliquas_z) > 0.3').shape[0]
    print 1.0*num/qsos.query('milliquas_z >= 0.0').shape[0]
    print "Milliquas photometric redshift outliers are about" ,100*num/qsos.query('photoz_rp >= 0.0').shape[0], "%"

    # Investigate objects missed by Color-z + RF method but selected via Milliquas
    # -> There is actually no single object only selected by Milliquas, interesting
    # cat['obs_class1'].replace(np.NaN,'None',inplace=True)
    # pzrf_sel = cat.query('(priority_rf == 0 and priority_photoz == 0) and obs_class1 == "QSO"')
    # print pzrf_sel[['obs_class1','priority','obs_z','obs_class2','obs_date','psfmag_i']]
    #
    # print cat.query('(priority_rf > 0 or priority_photoz > 0)').obs_class1.value_counts()
    # -> All objects that were not selected by Color-z + RF method were at higher psfmag_i
    # and therefore not selected naturally
    # -> If selection is allowed to extend to psfmag_i < 19.1, all objects are recovered between both methods


def eval_obs_obj(cat):

    # Prepare obs_class1 column
    cat['obs_class1'].replace(np.NaN,'None',inplace=True)
    cat.query('obs_class1 != "None"',inplace=True)

    # Total number of stuff
    print cat.obs_class1.value_counts()
    # Total number of quasars
    print cat.query('obs_class1 == "QSO"').obs_class2.value_counts()
    print "\n"


    # Exclude previously known objects
    cat = exclude_known_obj(cat)
    # Exclude already observed objects

    print "\n"
    print "NEWLY IDENTIFIED OBJECTS IN THIS CAMPAGIN"

    print cat.obs_class1.value_counts()

    print cat.query('obs_class1 == "QSO"').obs_class2.value_counts()

    print "\n"
    print "Number of QSOs with psfmag_i < 19.0 and z > 2.5 : "
    print cat.query('obs_class2 == "QSO" and psfmag_i < 19.0 and obs_z > 2.5').shape[0]
    print "\n"
    print "Number of QSOs with psfmag_i < 18.5 and z > 2.5 : "
    print cat.query('obs_class2 == "QSO" and psfmag_i < 18.5 and obs_z > 2.5').shape[0]
    print "\n"
    print "Number of QSOs with psfmag_i < 18.5 and z > 2.8 : "
    print cat.query('obs_class2 == "QSO" and psfmag_i < 18.5 and obs_z > 2.8').shape[0]
    print "\n"
    print "Number of QSOs with psfmag_i < 18.0 and z > 2.8 : "
    print cat.query('obs_class2 == "QSO" and psfmag_i < 18.0 and obs_z > 2.8').shape[0]
    print "\n"
    print "Number of BALs found"
    print cat.query('obs_class2 == "BAL"').shape[0]
    print "\n"
    print cat[['obs_class1','wise_designation','obs_class2','psfmag_i','obs_z']].query('obs_class1 == "QSO"').sort(columns=['obs_class2','obs_z','psfmag_i'])

    #   current contribution 2.6 - 5.7 % for 18.5 and 18.0 limit .... not a lot
