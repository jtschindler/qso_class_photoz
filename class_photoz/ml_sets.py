import pandas as pd
import numpy as np
import math
from sklearn import preprocessing, cross_validation


def build_matrices(df, features,label, drop_nans = True):

    """This routines returns the feature matrix X to use for the classification
    and the label vector y based on the input DataFrame. The label column must
    be df.label and the features must be valid column names of the DataFrame

    Input:
            df (DataFrame)
            features (list) list of label names to be considered
    Output:
            X (Numpy Array, 2D) feature matrix
            y (Numpy Array, 1D) label vector
    """

    if drop_nans:
        df.dropna(axis=0,how='any',subset=features,inplace=True)

    X = np.array(df[features])
    y = np.array(df[label])

    return X,y

def build_matrix(df, features,drop_nans = False):

    """This routines returns the feature matrix X to use for the classification.
    The features must be valid column names of the DataFrame.

    Input:
            df (DataFrame)
            features (list) list of label names to be considered
    Output:
            X (Numpy Array, 2D) feature matrix
    """

    if drop_nans:
        df.dropna(axis=0,how='any',subset=features,inplace=True)

    X = np.array(df[features])

    return X
