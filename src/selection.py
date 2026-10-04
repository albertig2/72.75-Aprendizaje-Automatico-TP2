import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, f_classif
from loading import load_data_set
from cleaning import clean_data_set
from preprocessing import split_features_target, make_preprocessor

def select_features(df, features, target):
    if features == "all":
        return df
    selected_features = features + [target]
    print(df[selected_features].head())
    return df[selected_features]

def correlation_matrix_plot(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def mutual_information_plot(X, Y, feature_names):

    mi_scores = mutual_info_classif(
        X,
        Y,
        random_state=42
    )

    mi_df = pd.DataFrame({
        "Feature": feature_names,
        "MI Score": mi_scores
    }).sort_values(
        "MI Score",
        ascending=False
    ).head(15)

    plt.figure(figsize=(10, 7))

    ax = sns.barplot(
        data=mi_df,
        x="MI Score",
        y="Feature"
    )

    ax.bar_label(
        ax.containers[0],
        fmt="%.3f",
        padding=3
    )

    plt.title("Top 15 Features by Mutual Information")
    plt.xlabel("MI Score")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.show()

def anova_f_test(X:np.ndarray, Y:np.ndarray):
    features = X
    target = Y
    f_scores, p_values = f_classif(features, target)

    results = pd.DataFrame({
        "Feature": features.columns,
        "F-score": f_scores,
        "p-value": p_values
    }).sort_values("F-score", ascending=False)

    print(results)


if __name__ == "__main__":
    data_raw = load_data_set()
    data_cleaned = clean_data_set(data_raw)
    X, Y = split_features_target(data_cleaned)
    preprocessor = make_preprocessor(X)
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()
    mutual_information_plot(X_processed, Y, feature_names)
    anova_f_test(X, Y)



    """
    contact	Drop	Telephone vs cellular is relatively low MI, and you don't seem interested in it
default	Drop	Very little contribution and lots of unknown; reasonable to exclude
month	Candidate to drop	Many dummy variables, individually low MI
day_of_week	Candidate to drop	Same argument; low individual MI
    
    
    
    """