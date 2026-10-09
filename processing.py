import pandas as pd


def prepare_input(user_inputs, feature_columns):
    """
    Convert user inputs into a DataFrame
    with the same column order used during training.
    """
    input_df = pd.DataFrame(
        [user_inputs],
        columns=feature_columns
    )

    return input_df