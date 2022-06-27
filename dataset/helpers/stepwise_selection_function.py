from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
import pandas as pd


class BackwardElimination:
    def __init__(self, data, target, scoring, tol=.05, cv=5):
        """
        Backward Elimination is a method for selecting features from a dataset. We are using
        the SequentialFeatureSelector from sklearn.feature_selection and the LogisticRegression as
        base estimator.

        :param
            data: pandas.DataFrame
            target: pandas.Series
            scoring: str
            tol: float
            cv: int

        """
        self.data = data
        self.target = target
        self.scoring = scoring
        self.tol = tol
        self.cv = cv
        self.selector_params = {
            'direction': 'backward',
            'n_features_to_select': 'auto',
            'scoring': self.scoring,
            'tol': self.tol,
            'cv': self.cv,
            'n_jobs': -1,
        }

        self.model = LogisticRegression(penalty='none', max_iter=500)
        self.selector = SequentialFeatureSelector(self.model, **self.selector_params)

    def fit(self):
        return self.selector.fit(self.data, self.target)

    def get_selected_columns(self):
        return self.data.columns[
            self.selector.get_support(indices=True)
        ].tolist()

    def get_transformed_data(self):
        return pd.DataFrame(
            self.selector.transform(self.data),
            columns=self.get_selected_columns()
        )

    def get_model(self):
        return self.model

    def get_selector(self):
        return self.selector


class ForwardSelection:
    def __init__(self, data, target, scoring, tol=0.05, cv=5):
        """
        Forward Selection is a method for selecting features from a dataset. We are using
        the SequentialFeatureSelector from sklearn.feature_selection and the LogisticRegression as
        base estimator.

        :param
            data: pandas.DataFrame
            target: pandas.Series
            scoring: str
            tol: float
            cv: int

        """
        self.data = data
        self.target = target
        self.scoring = scoring
        self.tol = tol
        self.cv = cv
        self.selector_params = {
            'direction': 'forward',
            'n_features_to_select': 'auto',
            'scoring': self.scoring,
            'tol': self.tol,
            'cv': self.cv,
            'n_jobs': -1,
        }

        self.model = LogisticRegression(penalty='none', max_iter=500)
        self.selector = SequentialFeatureSelector(self.model, **self.selector_params)

    def fit(self):
        return self.selector.fit(self.data, self.target)

    def get_selected_columns(self):
        return self.data.columns[
            self.selector.get_support(indices=True)
        ].tolist()

    def get_transformed_data(self):
        return pd.DataFrame(
            self.selector.transform(self.data),
            columns=self.get_selected_columns()
        )

    def get_model(self):
        return self.model

    def get_selector(self):
        return self.selector
