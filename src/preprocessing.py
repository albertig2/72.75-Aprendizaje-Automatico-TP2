from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split

def split_features_target(df):
    X = df.drop(columns=["y"])
    Y = df["y"].map({"no": 0, "yes": 1})

    return X, Y


def split_train_test(X, Y):
    return train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=Y
    )

def make_preprocessor(X:np.ndarray) -> ColumnTransformer:
    categorical_columns = X.select_dtypes(
        include=["object", "string"]
    ).columns

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns


    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numerical", StandardScaler(), numerical_columns),
        ]
    
    )
    return preprocessor