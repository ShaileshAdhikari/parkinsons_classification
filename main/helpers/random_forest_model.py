
import pandas as pd
import numpy as np
from helpers import get_train_test, RANDOM_STATE
from sklearn.tree import DecisionTreeClassifier
from scipy import stats

class RandomForestClassifierBagging:
    def __init__(self, n_estimators, criterion, max_depth, min_samples_split, class_weight, random_state:int):
        """
        Fit the model. This function is used to fit n_estimators of DecisionTree model.
        :param n_estimators: int
        :param criterion: str
        :param max_depth: int
        :param min_samples_split: int
        :param class_weight: int
        """
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.class_weight = class_weight
        self.accs = []
        self.clfs = []
        self.output = []
        
    def fit(self, X_train, y_train):
        """
        Fit the model. This function is used to fit n_estimators of DecisionTree model.
        :param X_train: pd.DataFrame
        :param y_train: pd.DataFrame
        :return: None
        """
        data = pd.concat([X_train, y_train], axis=1)
        for _ in range(self.n_estimators):
            df = data.sample(data.shape[0], replace=True)
            X = df.drop(['class'], axis=1)
            y = df['class']
            clf = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, class_weight=self.class_weight, min_samples_split=self.min_samples_split, random_state=self.random_state)
            clf.fit(X, y)
            self.clfs.append(clf)
        return
    
    def predict(self, X_test):
        """
        Predict. This function is used to predict for randomForest bagging model with given x_test input
        :param X: pd.DataFrame
        :return: np.ndarray
        """
        for i in range(self.n_estimators):
            self.output.append(self.clfs[i].predict(X_test))
#         print(self.output)
        mode, _ = stats.mode(self.output)
        return mode.reshape(-1)