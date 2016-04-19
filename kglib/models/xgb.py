import xgboost as xgb

from .base import Model


class XgbModel(Model):

    def __init__(self, **kwargs):
        kwargs.setdefault('silent', 1)
        self.num_boost_round = kwargs.pop('num_boost_round', 10)
        self._bst = None
        super().__init__(**kwargs)

    def _do_fit(self, X_train, y_train, X_test=None, y_test=None, **kwargs):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self._bst = xgb.train(kwargs, dtrain, self.num_boost_round)

    def _do_predict(self, X):
        data = xgb.DMatrix(X)
        y_pred = self._bst.predict(data)
        return y_pred
