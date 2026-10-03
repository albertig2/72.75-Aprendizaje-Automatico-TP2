from classifier import Classifier
from sklearn.model_selection import KFold
import numpy as np


def cross_validation(X: np.ndarray, Y: np.ndarray, model: Classifier, k: int) -> float:
    kf = KFold(n_split=k, shuffle=True, random_state=42)
    rmse = 0
    for _, (train_index, eval_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index]
        X_eval = X.iloc[eval_index]
        Y_train = Y.iloc[train_index]
        Y_eval = Y.iloc[eval_index]
    
        model.train()
        rmse += model.test() / k
    
    return rmse