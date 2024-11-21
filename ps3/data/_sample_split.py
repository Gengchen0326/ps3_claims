import pandas as pd
import hashlib

def create_sample_split(df: pd.DataFrame, id_column: str, training_frac: float = 0.8) -> pd.DataFrame:
    """
    Create a sample split column based on the specified ID column.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    id_column : str
        Name of the ID column to base the split on.
    training_frac : float, optional
        Fraction of data to assign to the training set, by default 0.8.

    Returns
    -------
    pd.DataFrame
        The dataframe with an additional column 'sample' indicating 'train' or 'test'.
    """
    # Ensure the ID column is of string type for hashing
    id_series = df[id_column].astype(str)

    # Use hashlib to generate deterministic integer values from the IDs
    hash_values = id_series.apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100)

    # Assign rows to 'train' or 'test' based on the hash value and training fraction
    df['sample'] = hash_values.apply(lambda x: 'train' if x < training_frac * 100 else 'test')
    
    return df

