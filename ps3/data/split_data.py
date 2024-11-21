import pandas as pd
import hashlib

def create_sample_split(df: pd.DataFrame, columns: list, train_fraction: float = 0.8) -> pd.DataFrame:
    """
    Create a sample column to split the dataframe into train and test sets deterministically.

    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - columns (list): The list of columns to base the split on (e.g., primary key).
    - train_fraction (float): Fraction of data to be assigned to the train set (default is 0.8).

    Returns:
    - pd.DataFrame: Dataframe with an additional column 'sample' indicating train or test.
    """
    # Concatenate the values of the specified columns to create a unique key for each row
    hash_values = df[columns].astype(str).sum(axis=1).apply(lambda x: int(hashlib.md5(x.encode()).hexdigest(), 16) % 100)
    
    # Assign rows to 'train' or 'test' based on the hash value and train_fraction
    df['sample'] = hash_values.apply(lambda x: 'train' if x < train_fraction * 100 else 'test')
    
    return df