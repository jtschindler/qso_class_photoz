import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn import preprocessing

from sklearn.svm import SVR

import ml_sets as sets
import ml_analysis as ml_an
import photoz_analysis as pz_an

def svm_reg_grid_search(df,features,label,param_grid,rand_state,scores,name):
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

            rand_state : integer
            Setting the random state variables to ensure reproducibility

            scores : list of strings
            Setting the score by which the grid search should be evaluated

            name : strings
            Setting the name of the output file for the grid search which
            contains all information about the grid

    """

    X,y = sets.build_matrices(df, features,label)

    # Standardizing the data
    X = preprocessing.robust_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2,random_state=rand_state)

    print "Training sample size: ", X_train.shape
    print "Evaluation sample size: ", X_test.shape

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        reg = GridSearchCV(SVR(), \
                        param_grid,scoring='%s' % score,cv=5,n_jobs=6)

        reg.fit(X_train, y_train)

        print("Best parameters set found on training set:")
        print()
        print(reg.best_params_)
        print()
        print("Grid scores on training set:")
        print()
        means = reg.cv_results_['mean_test_score']
        stds = reg.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, reg.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()
        df = pd.DataFrame(reg.cv_results_)
        df.to_hdf('SVR_GS_'+name+'_'+score+'.hdf5','data')
        print()
        print("The model is trained on the full development set (80%).")
        print("The scores are computed on the full evaluation set (20%).")
        print()
        y_true, y_pred = y_test, reg.predict(X_test)
        ml_an.evaluate_regression(y_test,y_pred)
        pz_an.evaluate_photoz(y_test,y_pred)
        print()

def svm_reg_example(df,features,label,params,rand_state):
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

            rand_state : integer
            Setting the random state variables to ensure reproducibility


    """

    # Building test and training sample
    X,y = sets.build_matrices(df, features, label)

    # Standardizing the data
    X = preprocessing.robust_scale(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=rand_state)


    # Random Forest Regression
    reg = SVR(**params)

    reg.fit(X_train,y_train)

    y_pred = reg.predict(X_test)



    # Evaluate regression method

    ml_an.evaluate_regression(y_test,y_pred)


    pz_an.plot_redshifts(y_test,y_pred)
    pz_an.plot_error_hist(y_test,y_pred)
    plt.show()

def svm_reg_predict(train_set, pred_set, features, label, params, pred_label):
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

    # Standardizing the data
    train_X = preprocessing.robust_scale(train_X)
    pred_X = preprocessing.robust_scale(pred_X)

    # Random Forest Regression
    reg = SVR(**params)
    reg.fit(train_X,train_y)

    pred_set[pred_label] = reg.predict(pred_X)

    return pred_set
