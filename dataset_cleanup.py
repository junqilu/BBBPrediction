import pandas as pd

from sklearn.feature_selection import \
    VarianceThreshold  # For checking descriptors with low variance

def read_tsv_to_df(tsv_file_directory):
    """
    Convert the .tsv file to pd df

    Args:
        tsv_file_directory (str): directory for the .tsv file

    Returns:
        df (pd df): df that contains the data from the .csv file

    """
    df = pd.read_csv(
        tsv_file_directory,
        sep='\t'  # .tsv files are similar to .csv files but instead of using
        # comma, .tsv files use tabs as separator
    )

    return df

def filter_low_variance_worker(intput_df, threshold_level):
    """
    Core function of filtering intput_df, using threshold_level given,
    out of columns with very low variance

    Args:
        intput_df (pd df): df to be filtered out of the low-variance col
        threshold_level (int): how much variance needed to pass the
        filtering. Default to 0 which means the column have all the values
        to be the same

    Returns:
        output_df (pd df): filtered df
    """
    feature_selector = VarianceThreshold(threshold=threshold_level)

    output_df = pd.DataFrame(
        feature_selector.fit_transform(intput_df),
        columns=intput_df.columns[
            feature_selector.get_support(indices=True)
            # This returns the indices of the kept descriptors
        ]
    )
    return output_df


def filter_low_variance(input_df, exclude_col_list,
                        threshold_level=0):
    """
    Filter out the columns in input_df, excluding columns in
    exclude_col_list, that have very low variances

    Args:
        input_df (pd df): df to be filtered based on low variance
        exclude_col_list (list of str): col in df that will be excluded from
        the filtering
        threshold_level (int): how much variance needed to pass the
        filtering. Default to 0 which means the column have all the values
        to be the same

    Returns:
        input_df_expanded_varianced (pd df): the df that have filtered out
        the low variance col
    """

    input_df.columns = input_df.columns.astype(str)  # Convert all column
    # titles into str. This is required by feature_selector .fit_transform

    input_feature_df = input_df.loc[
                       :,
                       ~input_df.columns.isin(exclude_col_list)
                       ]  # Extract out all the feature columns that need to go through the
    # low-variance filtering

    print(
        'Before removing the low-variance descriptors, the dataset has {} '
        'descriptors'.format(
            len(input_feature_df.columns)
        )
    )

    input_feature_df_varianced = filter_low_variance_worker(
        input_feature_df, threshold_level)

    input_df_expanded_varianced = pd.concat(  # Merge the filtered part back
        # to the original input_df with the excluded col
        [
            input_df[exclude_col_list],
            input_feature_df_varianced
        ],
        axis=1
    )

    print(
        'After removing the zero-variance descriptors, the dataset has {} '
        'descriptors'.format(
            len(input_df_expanded_varianced.columns)
        ),
    )

    return input_df_expanded_varianced
