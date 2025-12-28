import pandas as pd
import json

def inspect_metadata(df):
    """Returns shape, data types, and null counts."""
    info = {
        "shape": df.shape,
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict(),
        "null_counts": df.isnull().sum().to_dict()
    }
    return json.dumps(info)

def get_column_stats(df, col):
    """Returns distribution or unique values for a column."""
    if col not in df.columns:
        return f"Error: Column {col} not found"
    
    stats = {}
    if pd.api.types.is_numeric_dtype(df[col]):
        stats = df[col].describe().to_dict()
    else:
        stats = {
            "unique_values": df[col].nunique(),
            "top_values": df[col].value_counts().head(5).to_dict()
        }
    return json.dumps(stats)

def impute_missing(df, col, strategy):
    """Fills NaNs (mean, median, mode)."""
    if col not in df.columns:
        return df, f"Error: Column {col} not found"
    
    if strategy == "mean":
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median":
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        return df, f"Error: Unknown strategy {strategy}"
    
    return df, f"Imputed {col} using {strategy}"

def drop_column(df, col):
    """Removes unusable columns."""
    if col not in df.columns:
        return df, f"Error: Column {col} not found"
    
    df = df.drop(columns=[col])
    return df, f"Dropped column {col}"
