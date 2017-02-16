import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

import ml_sets as sets
import ml_analysis as ml_an

def rf_class_grid_search(df, features, label, param_grid, rand_state, scores, name):
    """This routine calculates the random forest classification on a
    grid of hyper-parameters for the random forest method to test the best
    support vector classification hyper-parameters. The results of the test
    will be written out.

    Input:
            df (DataFrame) The database to draw from
            features (list) list of features in the DataFrame

            label : string
            The label for the classification

    Output:
            None
    """

    X,y = sets.build_matrices(df, features,label)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=rand_state)

    print X_train.shape, X_test.shape

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(RandomForestClassifier(random_state=rand_state),
            param_grid, cv=5, scoring='%s' % score, n_jobs = 4)

        clf.fit(X_train, y_train)

        print("Detailed classification report:")
        print()
        print("The model is trained on the training set.")
        print("The scores are computed on the test set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

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
        df.to_hdf('RF_GS_CLASS_'+name+'_'+score+'.hdf5','data')
        print()
        print("The model is trained on the full development set (80%).")
        print("The scores are computed on the full evaluation set (20%).")
        print()
        y_true, y_pred = y_test, reg.predict(X_test)
        print()


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

    clf = RandomForestClassifier(**params)
    title = "Validation curve / Random Forest Classifier"
    ml_an.plot_validation_curve(clf, param_name, param_range, title, X, y,
                                            ylim=(0.0, 1.1), cv=None, n_jobs=4)

    plt.show()


def rf_class_example(df, features, label, params,rand_state):
    """This routine calculates an example of the random forest classification
     method. It prints the classification report and feature importances, the
     ROC AUC score and shows the learning curve for the chosen hyper-parameters
     as well as the ROC curve.

    Input:
            df (DataFrame) The database to draw from
            features (list) list of features in the DataFrame

            label : string
            The label for the regression

            params : dictionary
            List of input parameters for the classification

    Output:
            None
    """

    X,y = sets.build_matrices(df, features,label)

    # score curves, each time with 20% data randomly selected for validation.
    cv = cross_validation.ShuffleSplit(df.shape[0], n_iter=10,
                                   test_size=0.2, random_state=rand_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=rand_state)


    clf = RandomForestClassifier(**params)

    # ml_an.plot_learning_curve(clf, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
    # plt.show()

    clf.fit(X_train,y_train)

    y_true, y_pred = y_test, clf.predict(X_test)

    # Confusion matrix
    class_names = clf.classes_
    cnf_matrix = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
    plt.figure()

    ml_an.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')


    # y_pred_proba = clf.predict_proba(X_test)[:, 0]

    # ml_an.plot_precision_recall_curve(y_true,y_pred_proba,pos_label="QSO")
    # plt.show()

    feat_importances = clf.feature_importances_

    print "Classification Report "
    print(classification_report(y_true, y_pred))
    print "\n"
    print "Feature Importance "
    for i in range(len(features)):
        print str(features[i])+": "+str(feat_importances[i])
    print "\n"

    # ml_an.plot_roc_curve(y_true, y_pred_proba, pos_label="highz")

    plt.show()


def rf_class_predict(train_set, pred_set, features, params, class_label,
                    pred_label, class_0_label, class_1_label):

    """This function predicts the regression values for pred_set based on the
    features specified in the train_set

    Parameters:
          train_set : pandas dataframe
          The dataframe containing the features and the label for the
          classification.

          pred_set : pandas dataframe
          The dataframe containing the features for prediction

          features : list of strings
          List of features

          params : dictionary
          List of input parameters for the classification

          class_label : string
          The label for the classification

          pred_Label : string
          The name of the new column in pred_set for the classification results

          class_0_label : string
          The name of the new column in pred_set containing the class 0
          probabilities

          class_1_label : string
          The name of the new column in pred_set containing the class 1
          probabilities

    Output:
          pred_set : pandas dataframe
          The dataframe containing the features for prediction and the
          regression values in the pred_label named column.
    """

    train_X, train_y = sets.build_matrices(train_set, features,label=class_label)

    pred_X = sets.build_matrix(pred_set, features)

    clf = RandomForestClassifier(**params)

    clf.fit(train_X,train_y)

    pred_set[class_0_label] = clf.predict_proba(pred_X)[:, 0]
    pred_set[class_1_label] = clf.predict_proba(pred_X)[:, 1]
    pred_set[pred_label] = clf.predict(pred_X)

    return pred_set
