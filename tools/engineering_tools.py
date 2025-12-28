import pandas as pd
import json

def create_interaction(df, col1, col2, operation):
    """Creates a new column via math (e.g., df['income_per_age'] = df['income'] / df['age'])."""
    new_col = f"{col1}_{operation}_{col2}"
    if operation == "add":
        df[new_col] = df[col1] + df[col2]
    elif operation == "subtract":
        df[new_col] = df[col1] - df[col2]
    elif operation == "multiply":
        df[new_col] = df[col1] * df[col2]
    elif operation == "divide":
        df[new_col] = df[col1] / df[col2]
    else:
        return df, f"Error: Unknown operation {operation}"
    return df, f"Created interaction feature: {new_col}"

def encode_categorical(df, col, method="label"):
    """Applies One-Hot or Label encoding."""
    if col not in df.columns:
        return df, f"Error: Column {col} not found"
    
    if method == "label":
        df[col] = df[col].astype('category').cat.codes
    elif method == "onehot":
        df = pd.get_dummies(df, columns=[col], prefix=col)
    else:
        return df, f"Error: Unknown method {method}"
    
    return df, f"Encoded {col} using {method}"

def correlation_analysis(df, target):
    """Checks how features relate to the label."""
    if target not in df.columns:
        return f"Error: Target {target} not found"
    
    # Only numeric for correlation
    numeric_df = df.select_dtypes(include=['number'])
    if target not in numeric_df.columns:
        return f"Error: Target {target} is not numeric"
        
    corr = numeric_df.corr()[target].sort_values(ascending=False)
    return corr.to_json()

def select_top_features(df, target, k):
    """Keeps only the k most predictive features based on correlation."""
    if target not in df.columns:
        return df, f"Error: Target {target} not found"
    
    numeric_df = df.select_dtypes(include=['number'])
    corr = numeric_df.corr()[target].abs().sort_values(ascending=False)
    top_features = corr.head(k + 1).index.tolist() # +1 for target
    
    df = df[top_features]
    return df, f"Selected top {k} features: {', '.join(top_features)}"
