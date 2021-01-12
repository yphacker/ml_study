# coding=utf-8
# author=yphacker

import pandas as pd
import numpy as np
from collections import Counter


def get_num_object(df):
    features = df.columns.tolist()
    # features.remove('') # 去掉id和标签
    object_features = df.columns[df.dtypes == 'object'].tolist()
    num_features = list(set(features) - set(object_features))
    print('num_features:{}'.format(num_features))
    print('object_features:{}'.format(object_features))

    for object_feature in object_features:
        print('{}:{}'.format(object_feature, df[object_feature].value_counts().shape))


def check_null_columns(df):
    columns = df.columns.to_list()
    for column in columns:
        if df[column].isnull().mean() != 0:
            print(column, df[column].isnull().mean())


def detect_outliers(df, n, features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(k for k, v in outlier_indices.items() if v > n)

    return multiple_outliers

# outliers_to_drop = detect_outliers(train, 1, ['age', 'balance', 'duration'])
