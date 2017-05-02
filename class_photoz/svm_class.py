import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.learning_curve import validation_curve
from sklearn.learning_curve import learning_curve

from sklearn.svm import SVC

import ml_sets as sets
import ml_analysis as ml_an

def svm_grid_search(df,features,param_grid):
    """This routine calculates the support vector machine classification on a
    grid of hyper-parameters for the SVM method to test the best support vector
    classification hyper-parameters. The results of the test will be written
    out.

    Input:
            df (DataFrame) The database to draw from
            features (list) list of features in the DataFrame

    Output:
            None
    """

    X,y = sets.build_matrices(df, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=0)


    scores = ['precision_weighted', 'recall_weighted','f1_score']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(C=1), param_grid, cv=5,
                           scoring='%s' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on training set:")
        print()
        for params, mean_score, scores in clf.grid_scores_:
            print("%0.4f (+/-%0.04f) for %r"
                  % (mean_score, scores.std() * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the training set.")
        print("The scores are computed on the test set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()

def svm_validation_curve(df, features, params, param_name, param_range):
    """This routine calculates the validation curve for one hyper-parameter of
    the SVM classification method.

    Input:
            df (DataFrame) The database to draw from
            features (list) list of features in the DataFrame
            param_name (string) name of the hyper parameter
            param_range (list) list of parameter values to use

    Output:
            None
    """

    X,y = sets.build_matrices(df, features)

    clf = SVC(**params)
    title = "Validation curve / SVM classifier"
    ml_an.plot_validation_curve(clf, param_name, param_range, title, X, y,
                                        ylim=(0.0, 1.1), cv=None, n_jobs=4)

    plt.show()

def svm_example(df, features, params):
    """This routine calculates an example of the SVM classification method. It
    prints the classification report, the ROC AUC and shows the learning curve
    for the chosen hyper-parameters as well as the ROC curve.

    Input:
            df (DataFrame) The database to draw from
            features (list) list of features in the DataFrame

    Output:
            None
    """

    X,y = sets.build_matrices(df, features)

    # score curves, each time with 20% data randomly selected for validation.
    cv = cross_validation.ShuffleSplit(df.shape[0], n_iter=10,
                                   test_size=0.2, random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=0)

    clf = SVC(**params)


    title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$, C=10)"
    ml_an.plot_learning_curve(clf, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)
    plt.show()

    clf.fit(X_train,y_train)
    y_true, y_pred = y_test, clf.predict(X_test)

    print "Classification Report "
    print(classification_report(y_true, y_pred))
    print "\n"
    print "\n"
    print "Feature Importance "
    for i in range(len(features)):
        print str(features[i])+": "+str(feat_importances[i])
    print "\n"

    y_pred_rf = clf.predict_proba(X_test)[:, 0]

    ml_an.plot_precision_recall_curve(y_true,y_pred_proba,pos_label="QSO")
    plt.show()

    ml_an.plot_roc_curve(y_true, y_pred_proba, pos_label="QSO")

    plt.show()
