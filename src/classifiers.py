from sklearn import naive_bayes, neighbors, svm, ensemble
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import Pipeline

import numpy as np

class Classifier:
    def train(self, X_train:np.ndarray, Y_train:np.ndarray) -> None: 
            self.model.fit(X_train, Y_train)
    
    def test(self, X_test:np.ndarray, Y_test:np.ndarray) -> float: 
        Y_pred = self.model.predict(X_test)
        return accuracy_score(Y_test, Y_pred)
        #mse = mean_squared_error(Y_test, Y_pred)
        #rmse = np.sqrt(mse)
        #return rmse 

class NaiveBayes(Classifier):
    # Dont remember that much, but LDA, QDA functions where biggest discriminant function was the class.
    # Difference here from the other two is that the Covariace matrix C in LDA is shared and full, QDA not shared but full.
    # Here diagonal but not shared. 
    def __init__(self, preprocessor) -> None: 
        self.model = Pipeline([("preprocessing", preprocessor), ("classifier", naive_bayes.GaussianNB())])
        

class KNN(Classifier):
    # K-Nearest-Neighbor
    # The methods with weights git me a little confused, 
    # But we can use that points close are weigthed more, 
    # so that the boarders consider the closest points more
    # There where different methods to avoid checking the whole set of points per input
    # One with hyperplanes (k of them, and then how many buckets L and also k as hyperparameter)
    # Other method was using RF to split into buckets for comparison
    def __init__(self, preprocessor, k) -> None: 
        # I think I should have weights to be distance, (inverse of distance, but could be uniform also...)
        # Should find out how to do the this with the two types of limiting the search space
        self.model = Pipeline([("preprocessing", preprocessor), ("classifier", neighbors.KNeighborsClassifier(n_neighbors=k, weights='distance'))])
        # Initialize weights
        # Not sure of how
        # Eucledian distance (probably), but manhatten distance is an option
        # euclidean distance or manhetten?

class RandomForest(Classifier):
    # Random three turns into random forest,
    # where random threes generate some thresholds for all features
    # and takes the best threshold regarding a function
    # and splits in the feature with best threshold, 
    # hyperparameters: number of thresholds generated per class (if even per class) 
    # Could be generated where change of class (not sure of how)
    # Random forest does this, just each three uses a random set of the features
    # Splits like binary threes
    def __init__(self, preprocessor, max_depth, num_trees) -> None:
        self.model = Pipeline([("preprocessing", preprocessor), ("classifier", ensemble.RandomForestClassifier(n_estimators=num_trees, criterion='gini', max_depth=max_depth))])
        # self.model = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=max_depth, )
        # I think that splitter can be random, and that should try entropi criterion.
        # How to make it a forest(?)


class SVM(Classifier): 
    # support vector machine
    # This one I am a little confused by. 
    # There are closests points of each class are considered the principle vectors
    # Can be different amounts for each class
    # But in some function they are weighted differently
    # LDA, QDA (linear and quadratic gaussian, first assuming same diagonal and same C for all, 
    # second assuming same C but not diagonal) pluss anotherone
    # There should be a margin between the boareder between the classes
    # Can allow some misclassifications to increase margin
    # There exists different functions (non linear once, to avoid increasing dimentionality)
    # Every point could be taken away and we could be left with only the support vectors, 
    # we would still get the same classifier
    # C: how many misclassified samples we allow
    def __init__(self, preprocessor, kernel:str, C:int) -> None: 
        # kernel: should try linear, poly and rbf
        # LinearSVC fits the same linear model as SVC(kernel='linear') but scales to ~33k rows;
        # SVC with a kernel is O(n^2)-O(n^3) and very slow here.
        if kernel == 'linear':
            classifier = svm.LinearSVC(C=C)
        else:
            classifier = svm.SVC(kernel=kernel, C=C, cache_size=1000)
        self.model = Pipeline([("preprocessing", preprocessor), ("classifier", classifier)])