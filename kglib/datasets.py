import pandas as pd

from . import utils


class PandasDataset:
    """A wrapper for lazy loading pandas.DataFrame objects."""

    default_extension = ''

    def __init__(self, filename, *, dataframe=None):
        self.filename = utils.ensure_extension(filename, self.default_extension)
        self._dataframe = dataframe

    @property
    def dataframe(self):
        raise NotImplementedError

    def __str__(self):
        string = '{} at {}'.format(self.__class__.__name__, self.filename)
        return string

    @property
    def params(self):
        """Keyword arguments to reinstantiate the dataset."""
        return {}

    def copy_to(self, other):
        """Make another dataset a copy of this one."""
        raise NotImplementedError


class PandasCsvDataset(PandasDataset):

    default_extension = 'csv'

    def __init__(self, filename, *, read_csv_params=None, to_csv_params=None, dataframe=None):
        super().__init__(filename, dataframe=dataframe)
        self.read_csv_params = read_csv_params or {}
        self.to_csv_params = to_csv_params or {}

    @property
    def dataframe(self):
        if self._dataframe is None:
            self._dataframe = self._load_dataframe(self.filename, **self.read_csv_params)
        return self._dataframe

    @property
    def params(self):
        return {
            'read_csv_params': self.read_csv_params,
            'to_csv_params': self.to_csv_params,
        }

    def __str__(self):
        return self.filename

    @staticmethod
    def _load_dataframe(filename, **kwargs):
        return pd.read_csv(filename, **kwargs)

    @staticmethod
    def _save_dataframe(dataframe, filename, **kwargs):
        dataframe.to_csv(filename, **kwargs)

    def copy_to(self, other):
        other.read_csv_params = self.read_csv_params
        other.to_csv_params = self.to_csv_params
        other.dataframe = self.dataframe.copy()

    def save(self):
        self._save_dataframe(self.dataframe, self.filename, **self.to_csv_params)

    @classmethod
    def load(cls, filename, **kwargs):
        obj = cls(filename, **kwargs)
        return obj
