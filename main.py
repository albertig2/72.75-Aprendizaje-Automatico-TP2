from src.loading import load_data_set
from src.cleaning import clean_data_set
from src.selection import select_features
from src.preprocessing import make_preprocessor, split_features_target, split_train_test
from src.classifiers import NaiveBayes, KNN, RandomForest, SVM
from src.evaluation import cross_validate


df = load_data_set()
df_clean = clean_data_set(df)

A_drop = ["duration"]
B_drop = ["duration", "contact", "default"]
C_drop = [
    "contact",
    "default",
    "month",
    "day_of_week",
    "duration"
]

selected_features = [
    column for column in df_clean.columns
    if column not in A_drop
]

df_selected = select_features(df_clean, selected_features, "y")
X, Y = split_features_target(df_clean)
X_train, X_test, Y_train, Y_test = split_train_test(X, Y)

preprocessor = make_preprocessor(X_train)

nbc = NaiveBayes(preprocessor)
knnc = KNN(preprocessor, k=5)
rfc = RandomForest(preprocessor, max_depth=5, num_trees=50)
svmc = SVM(preprocessor, kernel='linear', C=10)

classifiers = [nbc, knnc, rfc, svmc]

for classifier in classifiers:
    scores = cross_validate(
        classifier.model,
        X,
        Y,
        folds=5
    )

    print("CV scores:", scores)
    print("Mean:", scores.mean())