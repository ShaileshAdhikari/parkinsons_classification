from typing import Any, List, Tuple, Dict

import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import confusion_matrix as confusion_mat
from collections import Counter
import xgboost as xgb


def pos_ratio(Y) -> float:
    """
    Compute the ratio of positive class in the dataset.
    """
    counter = Counter(Y)
    sum_wpos = counter[1]
    sum_wneg = counter[0]

    # print weight stat
    # print('weight statistics: wpos=%g, wneg=%g, ratio=%g' % (sum_wpos, sum_wneg, sum_wneg / sum_wpos))
    return sum_wneg / sum_wpos


def eval_metrics(cm1: np.ndarray, ) -> Dict:
    """
    Compute evaluation metric from Confusing matrix.
    :param cm1: confusion matrix
    :return: dict of evaluation
    """
    TP = cm1[1][1]
    TN = cm1[0][0]
    FN = cm1[1][0]
    FP = cm1[0][1]

    total1 = TP + TN + FP + FN
    accuracy1 = (TP + TN) / total1
    # How often actually positive are predicted as positive by our classifier?
    sensitivity1 = TP / (TP + FN)
    specificity1 = TN / (TN + FP)
    # How often predicted as positive by our classifier are actually positive?
    precision1 = TP / (TP + FP)
    # F1 score
    f1_score1 = 2 * (precision1 * sensitivity1) / (precision1 + sensitivity1)

    return {'accuracy': accuracy1,
            'sensitivity': sensitivity1,
            'specificity': specificity1,
            'precision': precision1,
            'f1_score': f1_score1}


class TrainXGBoost(object):
    """
    TrainXGBoost is a class for training XGBoost model.
    """

    def __init__(self,
                 **params: Any) -> None:
        self.reg_alpha = params.get("reg_alpha", 0.5)
        self.reg_lambda = params.get("reg_lambda", 0.5)
        self.max_depth = params.get("max_depth", 2)
        self.learning_rate = params.get("learning_rate", 0.1)
        self.gamma = params.get("gamma", 0)
        self.num_parallel_tree = params.get("num_parallel_tree", 4)
        self.min_child_weight = params.get("min_child_weight", 1)
        self.subsample = params.get("subsample", 0.8)
        self.colsample_bytree = params.get("colsample_bytree", 0.8)
        self.grow_policy = params.get("grow_policy", 'depthwise')
        self.booster = params.get("booster", 'gbtree')
        self.tree_method = params.get("tree_method", 'exact')

        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'tree_method': self.tree_method,
            'booster': self.booster,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'gamma': self.gamma,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'grow_policy': self.grow_policy,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'num_parallel_tree': self.num_parallel_tree
        }


    def set_params(self, **param):
        """
        Set the parameters of this estimator.
        :param param: Any
        :return: None
        """
        return self.__setattr__('params', param)

    def get_params(self, deep=True):
        """
        Get parameters of this estimator.
        :param deep: Boolean
        :return: Dictionary
        """
        # suppose this estimator has parameters "alpha" and "recursive"
        return self.params

    def train_test_preparation(
            self,
            train_x: DataFrame,
            train_y: DataFrame,
            test_x: DataFrame,
            test_y: DataFrame,
    ) -> Tuple[xgb.DMatrix, xgb.DMatrix]:
        """
        Train and test data preparation. This function is used to prepare train and test DMatrix data for XGBoost model.
        :return: Tuple[xgb.DMatrix, xgb.DMatrix]
        """

        self.params["scale_pos_weight"] = pos_ratio(train_y)
        # self.params['base_score'] = np.log(self.params["scale_pos_weight"])

        dm_train = xgb.DMatrix(train_x , label=train_y, feature_names=list(train_x.columns))
        dm_test = xgb.DMatrix(test_x , label=test_y, feature_names=list(test_x.columns))

        self.true_test_y = test_y
        self.true_train_y = train_y

        return dm_train, dm_test

    def run_cv(self, n_folds: int = 5,
               iterations: int = 100) -> DataFrame:
        """
        Run cross validation. This function is used to run cross validation for XGBoost model.
        :param n_folds: int
        :param iterations: int
        :return: DataFrame
        """

        print(f"XGBoost Cross-validation with {n_folds} folds ...")
        return xgb.cv(
            self.params,
            self.train,
            num_boost_round=iterations,
            nfold=n_folds,
            early_stopping_rounds=20,
            as_pandas=True,
            verbose_eval=True
        )

    def fit(self,
            X_train, y_train, X_test, y_test,
            xgb_model: xgb.Booster = None,
            iterations: int = 500,
            verbose: bool = False) -> xgb.Booster:
        """
        Fit the model. This function is used to fit XGBoost model.
        :param xgb_model: xgb.Booster
        :param iterations: int
        :param verbose: Boolean
        :return: xgb.Booster
        """
        self.train, self.test = self.train_test_preparation(X_train, y_train, X_test, y_test)

        print("Training XGBoost model...")
        self.model = xgb.train(
            self.params,
            self.train,
            num_boost_round=iterations,
            evals=[(self.train, 'Train'), (self.test, 'Test')],
            early_stopping_rounds=20,
            xgb_model=xgb_model,
            verbose_eval=verbose
        )

        return self.model

    def eval_model(self) -> Dict:
        """
        Evaluate the model. This function is used to evaluate XGBoost model.
        :param verbose: bool default False
        :return: Dictionary
        """

        self.prediction_for_testing = self.model.predict(self.test)
        self.confusion_matrix = confusion_mat(self.true_test_y.values,
                                              self.prediction_for_testing.round())
        return eval_metrics(self.confusion_matrix)
        # return eval_metrics(self.confusion_matrix)['accuracy']

    def _base_predict(self, X):
        return self.model.predict(X)

    def predict_proba(self,
                      X: xgb.DMatrix) -> np.ndarray:
        """
        Predict probabilities. This function is used to predict probabilities for XGBoost model.
        :param X: pd.DataFrame
        :return: np.ndarray
        """
        vstack = np.vstack
        classone_probs = self._base_predict(X)
        classzero_probs = 1.0 - classone_probs
        return vstack((classzero_probs, classone_probs)).transpose()

    def predict(self,
                X: pd.DataFrame,
                threshold: float = 0.5) -> np.ndarray:
        """
        Predict. This function is used to predict for XGBoost model with defined threshold.
        :param X: pd.DataFrame
        :param threshold: float, default 0.5
        :return: np.ndarray
        """
        x = xgb.DMatrix(X, feature_names=list(X.columns))

        return (self.predict_proba(x)[:, 1] >= threshold).astype(int)

    def score(self) -> float:
        """
        Score. This function is used to score XGBoost model.
        :return: float
        """
        return self.model.attributes()['best_score']

