import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split

import ml_analysis as ml_an
import photoz_analysis as pz_an

def gb_reg_example(df,features,label):
        """This routine calculates an example of the gradient boosting
        regression tuned to photometric redshift estimation.
        The results will be analyzed with the analyis routines/functions
        provided in ml_eval.py and photoz_analysis.py

        Parameters:
                df : pandas dataframe
                The dataframe containing the features and the label for the
                regression.

                features : list of strings
                List of features

                label : string
                The label for the regression
        """
    # Building test and training sample
    # X,y = sets.build_matrices(df, features, label)
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,y, test_size=0.2, random_state=0)
    #
    #
    # # Gradient Boosting Regressor
    # params = {'n_estimators': 50, 'max_depth': 25, 'min_samples_split': 2,
    #           'learning_rate': 0.1, 'loss': 'ls'}
    # reg = GradientBoostingRegressor(**params)
    #
    #
    # reg.fit(X_train,y_train)
    #
    # y_pred = reg.predict(X_test)
    #
    # feat_importances = reg.feature_importances_
    #
    #
    # # Evaluate regression method
    # print "Feature Importances "
    # for i in range(len(features)):
    #     print str(features[i])+": "+str(feat_importances[i])
    # print "\n"
    #
    # ml_an.evaluate_regression(y_test,y_pred)
    #
    # pz_an.evaluate_photoz(y_test,y_pred)
