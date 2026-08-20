import numpy as np
import pandas as pd
import argparse
from data_utils import Preprocessor, CyclicEncoder, datasets
import itertools


def metaSynthHyacinth(hierarchical_feats, df):
    df_meta = df[hierarchical_feats]
    unique_values = {col: sorted(df_meta[col].unique()) for col in df_meta.columns}
    combinations = list(itertools.product(*unique_values.values()))
    hierarchical_df = pd.DataFrame(combinations, columns=unique_values.keys())
    merged_df = hierarchical_df.merge(df, how='outer', on=hierarchical_feats, indicator=True)
    return merged_df


def metaSynthTimeWeaver(constraints, hierarchical_feats, df):
    df_meta = df[hierarchical_feats]
    unique_values = {col: sorted(df_meta[col].unique()) for col in df_meta.columns if col not in constraints}
    for key in constraints.keys():
        unique_values[key] = [constraints[key]]
    combinations = list(itertools.product(*unique_values.values()))
    hierarchical_df = pd.DataFrame(combinations, columns=unique_values.keys())
    for column in df.columns:
        if column not in hierarchical_feats:
            hierarchical_df[column] = np.NAN
    return hierarchical_df


def fetchSSSDTrainingMask(df=None, metadata_indices=None, ratio=0.5):
    print()

    print()


def metadataMask(metadata, synthmask, dataset):
    if synthmask not in ['C', 'M', 'F']:
        percentage = float(synthmask)
        n_rows = len(metadata)
        # Number of True values (25%)
        n_true = int(n_rows * percentage)
        # Randomly choose indices for True values
        true_indices = np.random.choice(metadata.index, size=n_true, replace=False)
        # Create the boolean Series
        bool_series = pd.Series(False, index=metadata.index)
        bool_series.loc[true_indices] = True
        return bool_series

    else:
        if dataset == "MetroTraffic":
            if synthmask == "C":
                return metadata['year'] == 2018
            elif synthmask == "M":
                return (metadata['year'] == 2018) & (metadata['day'] == 15)
            elif synthmask == "F":
                return (metadata['year'] == 2018) & (metadata['hour'] == 6)

        elif dataset == "AustraliaTourism":
            if synthmask == "C":
                return metadata['year'] == 2016
            elif synthmask == "M":
                return (metadata['year'] == 2016) & (metadata['State'] == 'Queensland')
            elif synthmask == 'F':
                return (metadata['year'] == 2016) & (metadata['Purpose'] == 'Holiday')

        elif dataset == 'BeijingAirQuality':
            if synthmask == "C":
                return metadata['year'] == 2017
            elif synthmask == "M":
                return (metadata['year'] == 2017) & (metadata['month'] == 2)
            elif synthmask == "F":
                return (metadata['year'] == 2017) & (metadata['hour'] == 11)

        elif dataset == 'RossmanSales':
            if synthmask == "C":
                return metadata['Year'] == 2015
            elif synthmask == "M":
                return (metadata['Year'] == 2015) & (metadata['Month'] == 3)
            elif synthmask == "F":
                return (metadata['Year'] == 2015) & (metadata['Store'] == 9)

        elif dataset == "PanamaEnergy":
            if synthmask == "C":
                return metadata['year'] == 2020
            elif synthmask == "M":
                return (metadata['year'] == 2020) & (metadata['day'] == 5)
            elif synthmask == "F":
                return (metadata['year'] == 2020) & (metadata['city'] == 'san')


        elif dataset == "QECParityZZ":
            # TODO fill with values that make sense
            if synthmask == "C":
                return metadata['shot_id'] <= 2000
            else:
                raise NotImplementedError('TODO define metadata properly')
