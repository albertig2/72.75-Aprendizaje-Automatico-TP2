from sklearn.model_selection import cross_val_score


def cross_validate(model, X, Y, folds=5):

    scores = cross_val_score(
        model,
        X,
        Y,
        cv=folds,
        scoring="accuracy"
    )

    return scores