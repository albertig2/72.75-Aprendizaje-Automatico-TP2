import sklearn
from typing import Protocol


class Classifier(Protocol):
    def train(self) -> None: ...


class NaiveBayes:
    # Dont remember that much, but LDA, QDA functions where biggest determinant function was the class
    def __init__(self) -> None: 
        pass
    
    def train(self) -> None: 
        pass

class KNN:
    # K-Nearest-Neighbor
    # The methods with weights git me a little confused, 
    # But we can use that points close are weigthed more, 
    # so that the boarders consider the closest points more
    # There where different methods to avoid checking the whole set of points per input
    # One with hyperplanes (k of them, and then how many buckets L and also k as hyperparameter)
    # Other method was using RF to split into buckets for comparison
    def __init__(self, k) -> None: 
        self.k = k
        self.W = [] # Weights
        
        # Initialize weights
        # Not sure of how
        # Eucledian distance (probably), but manhatten distance is an option
    
    def train(self) -> None:
        pass
        
class RandomForest:
    # Random three turns into random forest,
    # where random threes generate some thresholds for all features
    # and takes the best threshold regarding a function
    # and splits in the feature with best threshold, 
    # hyperparameters: number of thresholds generated per class (if even per class) 
    # Could be generated where change of class (not sure of how)
    # Random forest does this, just each three uses a random set of the features
    # Splits like binary threes
    def __init__(self) -> None:
        pass
    
    def train(self) -> None:
        pass

class SVM: 
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
    def __init__(self) -> None: 
            pass
        
    def train(self) -> None: 
        pass