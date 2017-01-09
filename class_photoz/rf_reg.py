import numpy as np
import matplotlib.pyplot as plt

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
# from sklearn.learning_curve import validation_curve
from sklearn import preprocessing

import ml_sets as sets
import ml_analysis as ml_an
import photoz_analysis as pz_an

def rf_reg_grid_search(df,features,label,param_grid):
    """This routine calculates the random forest regression on a grid of
    hyper-parameters for the random forest method to test the best
    hyper-parameters. The analysis results of the test will be written out.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            param_grid : dictionary-like structure
            Parameter grid of input parameters for the grid search

    """

    X,y = sets.build_matrices(df, features,label)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2,random_state=0)

    print X_train.shape, X_test.shape


    scores = ['r2']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        reg = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, cv=5,
                           scoring='%s' % score,n_jobs =4)
        reg.fit(X_train, y_train)

        print("Best parameters set found on training set:")
        print()
        print(reg.best_params_)
        print()
        print("Grid scores on training set:")
        print()
        for params, mean_score, scores in reg.grid_scores_:
            print("%0.6f (+/-%0.06f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()


def rf_reg_example(df,features,label,params):
    """This routine calculates an example of the random forest regression tuned
    to photometric redshift estimation. The results will be analyzed with the
    analyis routines/functions provided in ml_eval.py and photoz_analysis.py

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the regression

    """

    # Building test and training sample
    X,y = sets.build_matrices(df, features, label)

    # Standardizing the data -> Does not do much!
    # X = preprocessing.scale(X)
    # X = preprocessing.robust_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=0)

    # Leftover from trying out weights
    # w_train = X_train[:,-1]
    # X_train = X_train[:,:-1]
    # w_test = X_test[:,-1]
    # X_test = X_test[:,:-1]

    # Random Forest Regression
    reg = RandomForestRegressor(**params)

    reg.fit(X_train,y_train)

    y_pred = reg.predict(X_test)

    feat_importances = reg.feature_importances_


    # Evaluate regression method
    print "Feature Importances "
    for i in range(len(features)):
        print str(features[i])+": "+str(feat_importances[i])
    print "\n"

    ml_an.evaluate_regression(y_test,y_pred)


    pz_an.plot_redshifts(y_test,y_pred)
    pz_an.plot_error_hist(y_test,y_pred)
    plt.show()


def rf_reg_validation_curve(df,features,label,params,val_param,val_range):
    """This routine calculates the validation curve for random forest
    regression.

    Parameters:
            df : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            features : list of strings
            List of features

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the regression

            val_param : string
            Name of the validation parameter

            val_range : array-like
            List of parameter values for the validation curve

    """

    X,y = sets.build_matrices(df, features,label)

    # Random Forest Regression
    reg = RandomForestRegressor(**params)

    #Calculate and plot validation curve
    pz_an.plot_validation_curve(reg, val_param, val_range, X, y,
                                        ylim=(0.0, 1.1), cv=None, n_jobs=4)

    plt.show()


def rf_reg_predict(train_set, pred_set, features, label, params, pred_label):
    """This function predicts the regression values for pred_set based on the
    features specified in the train_set

    Parameters:
            train_set : pandas dataframe
            The dataframe containing the features and the label for the
            regression.

            pred_set : pandas dataframe
            The dataframe containing the features for prediction

            features : list of strings
            List of features

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the regression

            pred_label : string
            Name of the new label in the pred_set dataframe in which the
            predicted values are written

    Output:
            pred_set : pandas dataframe
            The dataframe containing the features for prediction and the
            regression values in the pred_label named column.
    """

    for feature in features:
      train_set.dropna(axis=0,how='any',subset=[feature],inplace=True)

    # Building test and training sample
    train_X, train_y = sets.build_matrices(train_set, features, label)

    pred_X = sets.build_matrix(pred_set, features)

    # Random Forest Regression
    reg = RandomForestRegressor(**params)
    reg.fit(train_X,train_y)

    pred_set[pred_label] = reg.predict(pred_X)

    return pred_set
