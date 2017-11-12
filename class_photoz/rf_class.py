
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import ml_sets as sets
import ml_analysis as ml_an


def rf_class_grid_search(df_train, df_pred, features, label, param_grid, rand_state, scores, name):
    """This routine calculates the random forest classification on a grid of
    hyper-parameters for the random forest method to test the best
    hyper-parameters. The analysis results of the test will be written out and
    saved.

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

    X_train, y_train = sets.build_matrices(df_train, features, label=label)
    X_test, y_test = sets.build_matrices(df_pred, features, label=label)

    print X_train.shape, X_test.shape

    print pd.Series(y_train).value_counts(), pd.Series(y_test).value_counts()

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(RandomForestClassifier(random_state=rand_state),
            param_grid, cv=5, scoring='%s' % score, n_jobs=4)

        clf.fit(X_train, y_train)

        print("Detailed classification report:")
        print()
        print("The model is trained on the training set.")
        print("The scores are computed on the test set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        y_true = y_true.astype('string')
        y_pred = y_pred.astype('string')

        print(classification_report(y_true, y_pred))
        print "\n"

        print("Best parameters set found on training set:\n")
        print(clf.best_params_)
        print "\n"
        print("Grid scores on training set:")
        print "\n"
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print "\n"
        df = pd.DataFrame(clf.cv_results_)
        df.to_hdf('RF_GS_CLASS_'+name+'_'+score+'.hdf5','data')


def rf_class_validation_curve(df, features, label, params, param_name, param_range):
    """This routine calculates the validation curve for one hyper-parameter of
    the random forest classification method.

    Input:
            df (DataFrame) The database to draw from
            features (list) list of features in the DataFrame

            label : string
            The label for the regression

            param_name (string) name of the hyper parameter
            param_range (list) list of parameter values to use


    Output:
            None
    """

    X,y = sets.build_matrices(df, features,label)

    # Standardizing the data
    # X = preprocessing.robust_scale(X)

    clf = RandomForestClassifier(**params)
    title = "Validation curve / Random Forest Classifier"
    ml_an.plot_validation_curve(clf, param_name, param_range, title, X, y,
                                            ylim=(0.0, 1.1), cv=None, n_jobs=4)

    plt.show()


def rf_class_predict(df_train, df_pred, features, label, params,
    rand_state):
    """This routine calculates an example of the random forest classification
     method. It is aimed at multi-class classification.
     It prints the classification report and feature importances and shows the
     confusion matrix for all classes.

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

    Return :
            clf : scikit-learn Classifier
            The Classifier trained on the training set

            y_pred : array-like
            An array with the predicted classes from df_pred

            y_prob : array-like
            An array with the predicted class probabilities
    """

    X_train, y_train = sets.build_matrices(df_train, features, label=label)
    X_pred = sets.build_matrix(df_pred, features)

    # Standardizing the data
    # X_train = preprocessing.robust_scale(X_train)
    # X_pred = preprocessing.robust_scale(X_pred)

    clf = RandomForestClassifier(**params)

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_pred)

    # Predicting the probabilities for the classes
    y_prob = clf.predict_proba(X_pred)

    return clf, y_pred, y_prob


def rf_class_example(df_train, df_pred, features, label, params, rand_state):
    """This routine calculates an example of the random forest classification
     method. It is aimed at multi-class classification.
     It prints the classification report and feature importances and shows the
     confusion matrix for all classes.

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

    clf, y_pred, y_prob = rf_class_predict(df_train,df_pred, features, label,
                                                            params, rand_state)

    X_pred, y_true = sets.build_matrices(df_pred, features,label=label)

    y_true = y_true.astype('string')
    y_pred = y_pred.astype('string')

    feat_importances = clf.feature_importances_

    print "Classification Report "
    print(classification_report(y_true, y_pred))
    print "\n"
    print "Feature Importance "
    for i in range(len(features)):
        print str(features[i])+": "+str(feat_importances[i])
    print "\n"

    # Confusion matrix
    class_names = clf.classes_
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)

    ml_an.my_confusion_matrix(cnf_matrix, classes=class_names)

    plt.show()

    # Predicting the probabilities for the classes
    y_prob = clf.predict_proba(X_pred)

    df_prob = pd.DataFrame(y_prob)
    df_prob.columns = clf.classes_
    df_prob.index = df_pred.index
    df_prob['qso_prob'] = df_prob.highz + df_prob.midz + df_prob.lowz + df_prob.vlowz
    df_prob['true_class'] = y_true
    df_prob['pred_class'] = y_pred

    return y_true, y_pred, df_prob
