import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing


import ml_sets as sets
import ml_analysis as ml_an


def prepare_qso_star_data(df_stars,df_qsos,features,label,rand_state):

    X_stars,y_stars = sets.build_matrices(df_stars, features,label)

    X_st_tr,X_st_te,y_st_tr,y_st_te = train_test_split(
            X_stars,y_stars, test_size=0.2,random_state=rand_state)

    X_qsos,y_qsos = sets.build_matrices(df_qsos, features,label)

    X_qs_tr,X_qs_te,y_qs_tr,y_qs_te = train_test_split(
            X_qsos,y_qsos, test_size=0.2,random_state=rand_state)

    X_train = np.concatenate((X_st_tr,X_qs_tr),axis=0)
    X_test = np.concatenate((X_st_te,X_qs_te),axis=0)

    y_train = np.concatenate((y_st_tr,y_qs_tr),axis=0)
    y_test = np.concatenate((y_st_te,y_qs_te),axis=0)

    # Standardizing the data
    X_train = preprocessing.robust_scale(X_train)
    X_test = preprocessing.robust_scale(X_test)


    return X_train,y_train,X_test,y_test

def rf_class_grid_search(df_train,df_pred, features, label, param_grid, rand_state, scores, name):
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


    X_train, y_train = sets.build_matrices(df_train, features,label=label)
    X_test, y_test = sets.build_matrices(df_pred, features,label=label)

    print X_train.shape, X_test.shape

    print pd.Series(y_train).value_counts() , pd.Series(y_test).value_counts()

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(RandomForestClassifier(random_state=rand_state),
            param_grid, cv=5, scoring='%s' % score, n_jobs = 3)

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
        print()

        print("Best parameters set found on training set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on training set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()
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

    print "THIS FUNCTION IS DEPRECATED"

    X,y = sets.build_matrices(df, features,label)

    # Standardizing the data
    X = preprocessing.robust_scale(X)

    clf = RandomForestClassifier(**params)
    title = "Validation curve / Random Forest Classifier"
    ml_an.plot_validation_curve(clf, param_name, param_range, title, X, y,
                                            ylim=(0.0, 1.1), cv=None, n_jobs=4)

    plt.show()


def  rf_class_predict(df_train, df_pred, features, label, params,
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
    """

    X_train, y_train = sets.build_matrices(df_train, features,label=label)
    X_pred = sets.build_matrix(df_pred, features)

    # Standardizing the data
    X_train = preprocessing.robust_scale(X_train)
    X_pred = preprocessing.robust_scale(X_pred)

    clf = RandomForestClassifier(**params)

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_pred)

    return clf, y_pred


# def rf_binaryclass_example(df_stars,df_qsos, features, label, params,rand_state):
#     """This routine calculates an example of the random forest classification
#      method. It is aimed at classification with only two classes (STAR/QSO).
#      It prints the classification report and feature importances, the
#      ROC/AUC score and shows the ROC curve and precision recall curve for the
#      chosen hyper-parameters.
#
#     Parameters:
#             df : pandas dataframe
#             The dataframe containing the features and the label for the
#             regression.
#
#             features : list of strings
#             List of features
#
#             label : string
#             The label for the regression
#
#             params : dictionary
#             List of input parameters for the regression
#
#             rand_state : integer
#             Setting the random state variables to ensure reproducibility
#     """
#
#     X_train, y_train, X_test, y_test = \
#             prepare_qso_star_data(df_stars,df_qsos,features,label,rand_state)
#
#     clf = RandomForestClassifier(**params)
#
#     clf.fit(X_train,y_train)
#
#     y_true, y_pred = y_test, clf.predict(X_test)
#
#     y_pred_proba = clf.predict_proba(X_test)[:, 0]
#
#     ml_an.plot_precision_recall_curve(y_true,y_pred_proba,pos_label="QSO")
#     plt.show()
#
#     feat_importances = clf.feature_importances_
#
#     print "Classification Report "
#     print(classification_report(y_true, y_pred))
#     print "\n"
#     print "Feature Importance "
#     for i in range(len(features)):
#         print str(features[i])+": "+str(feat_importances[i])
#     print "\n"
#
#     ml_an.plot_roc_curve(y_true, y_pred_proba, pos_label="QSO")
#
#     plt.show()



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

    clf, y_pred = rf_class_predict(df_train,df_pred, features, label,
                                                            params, rand_state)

    X_pred, y_true = sets.build_matrices(df_pred, features,label=label)


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


    ml_an.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')


    plt.show()





#
# def rf_class_predict(train_set, pred_set, features, params, class_label,
#                     pred_label, class_0_label, class_1_label):
#
#     """This function predicts the regression values for pred_set based on the
#     features specified in the train_set
#
#     Parameters:
#           train_set : pandas dataframe
#           The dataframe containing the features and the label for the
#           classification.
#
#           pred_set : pandas dataframe
#           The dataframe containing the features for prediction
#
#           features : list of strings
#           List of features
#
#           params : dictionary
#           List of input parameters for the classification
#
#           class_label : string
#           The label for the classification
#
#           pred_Label : string
#           The name of the new column in pred_set for the classification results
#
#           class_0_label : string
#           The name of the new column in pred_set containing the class 0
#           probabilities
#
#           class_1_label : string
#           The name of the new column in pred_set containing the class 1
#           probabilities
#
#     Output:
#           pred_set : pandas dataframe
#           The dataframe containing the features for prediction and the
#           regression values in the pred_label named column.
#     """
#
#     train_X, train_y = sets.build_matrices(train_set, features,label=class_label)
#
#     pred_X = sets.build_matrix(pred_set, features)
#
#     # Standardizing the data
#     train_X = preprocessing.robust_scale(train_X)
#     test_X = preprocessing.robust_scale(test_X)
#
#     # Random Forest Classification
#     clf = RandomForestClassifier(**params)
#     clf.fit(train_X,train_y)
#
#     pred_set[class_0_label] = clf.predict_proba(pred_X)[:, 0]
#     pred_set[class_1_label] = clf.predict_proba(pred_X)[:, 1]
#     pred_set[pred_label] = clf.predict(pred_X)
#
#     return pred_set
