import pickle

import pandas as pd


class Model:

    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, target_col, train_df, test_df=None):
        X_train, y_train = train_df.drop(target_col, axis=1).values, train_df[target_col].values

        if test_df is not None:
            X_test, y_test = test_df.drop(target_col, axis=1).values, test_df[target_col].values
        else:
            X_test, y_test = None, None

        self._do_fit(X_train, y_train, X_test, y_test, **self.params)

    def _do_fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        values = self._do_predict(X.values)
        series = pd.Series(values, index=X.index)
        return series

    def _do_predict(self, X):
        raise NotImplementedError

    def save(self, path):
        with open(path, 'wb') as dst:
            pickle.dump(self, dst)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as src:
            model = pickle.load(src)
        return model
