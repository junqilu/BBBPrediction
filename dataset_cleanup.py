import pandas as pd

from sklearn.feature_selection import \
    VarianceThreshold  # For checking descriptors with low variance
from collinearity import SelectNonCollinear  # For throw out highly


# correlated variables



def filter_low_variance(input_df, exclude_col_list,
                        threshold_level=0):
    feature_selector = VarianceThreshold(
        # feature_selector is for getting rid of low-variance features
        threshold=threshold_level
        # Variance = 0 means to get rid of features with all the same
        # values
    )

    input_feature_df = input_df.loc[
                       :,
                       ~input_df.columns.isin(exclude_col_list)
                       ]  # Extract
    # out all the feature columns that need to go through the low-variance filtering

    print(
        'Before removing the low-variance descriptors, the dataset has {} '
        'descriptors'.format(
            len(input_feature_df.columns)
        )
    )

    input_df.columns = input_df.columns.astype(
        str)  # Convert all column titles into str. This is required by
    # feature_selector .fit_transform

    feature_selector.fit_transform(input_feature_df)

    input_features_df_varianced = pd.DataFrame(
        feature_selector.fit_transform(input_feature_df),
        columns=input_feature_df.columns[
            feature_selector.get_support(indices=True)
            # This returns the indices of the kept descriptors
        ]
    )

    input_df_expanded_varianced = pd.concat(
        [input_df[exclude_col_list],
        input_features_df_varianced],
        axis=1
    )

    print(
        'After removing the zero-variance descriptors, the dataset has {} '
        'descriptors'.format(
            len(input_df_expanded_varianced.columns)
        ),
    )

    return input_df_expanded_varianced
