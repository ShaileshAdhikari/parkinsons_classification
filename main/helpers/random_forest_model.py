import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from scipy import stats


class RandomForestClassifierBagging:
    def __init__(
            self,
            class_weight: str = 'balanced',
            random_state: int = 42,
            **params
    ) -> None:
        """
        Fit the model. This function is used to fit n_estimators of DecisionTree model.
        :param
            **params:
                criterion : str, optional (default=’gini’)
                n_estimators : int, optional (default=10)
                max_depth : int or None, optional (default=None)
                min_samples_split : int, optional (default=2)
                splitter : str, optional (default=’best’)
                min_impurity_decrease : float, optional (default=0.0)
                ccp_alpha : float, optional (default=0.0)
                max_features : int or None, optional (default=None)
                min_samples_leaf : int, optional (default=1)
            class_weight: str, (default=’balanced’)
            random_state: int, (default=42)
        """
        self.criterion = params.get("criterion", "gini")
        self.n_estimators = params.get("n_estimators", 10)
        self.max_depth = params.get("max_depth", None)
        self.min_samples_split = params.get("min_samples_split", 2)
        self.splitter = params.get("splitter", "best")
        self.min_impurity_decrease = params.get("min_impurity_decrease", 0.0)
        self.ccp_alpha = params.get("ccp_alpha", 0.0)
        self.max_features = params.get("max_features", "auto")
        self.min_samples_leaf = params.get("min_sample_leaf", 1)
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
            df = data.sample(frac=0.6, replace=True, ignore_index=True, random_state=self.random_state)
            X = df.drop(['class'], axis=1)
            y = df['class']
            clf = DecisionTreeClassifier(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease,
                ccp_alpha=self.ccp_alpha,
                max_features=self.max_features,
                min_samples_leaf=self.min_samples_leaf,
                splitter=self.splitter,
                class_weight=self.class_weight,
                random_state=self.random_state
            )
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
