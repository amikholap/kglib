import datetime
import json
import os

import dateutil.parser

from . import utils


class Meta:
    """A key-value storage for metainformation."""

    VERSION = 1

    def __init__(self, filename):
        self._filename = filename
        self._parent_dir = os.path.dirname(self._filename)

        # Read from file or create it if it doesn't exist.
        if os.path.exists(self._filename):
            self._load()
        else:
            self._data = self._get_initial_data()

    @property
    def filename(self):
        return self._filename

    @property
    def directory(self):
        return os.path.dirname(self._filename)

    def __getattr__(self, key):
        try:
            attr = self._data[key]
        except KeyError:
            attr = super().__getattr__(key)
        return attr

    @classmethod
    def _to_json(cls, data):
        json_data = {}
        for key in ('created_at', 'updated_at'):
            json_data[key] = data[key].isoformat()
        json_data['version'] = data['version']
        return json_data

    # pylint: disable=unused-variable
    @classmethod
    def _from_json(cls, json_data, *, parent_dir):
        data = {}
        for key in ('created_at', 'updated_at'):
            data[key] = dateutil.parser.parse(json_data[key])
        data['version'] = json_data['version']
        return data

    @classmethod
    def _get_initial_data(cls):
        now = datetime.datetime.now()
        data = {
            'created_at': now,
            'updated_at': now,
            'version': cls.VERSION,
        }
        return data

    def _load(self):
        with open(self._filename) as src:
            json_data = json.load(src)
        self._data = self._from_json(json_data, parent_dir=self._parent_dir)
        assert self._data.keys() == json_data.keys()

    def save(self):
        json_data = self._to_json(self._data)
        assert self._data.keys() == json_data.keys()
        with open(self._filename, 'w') as dst:
            json.dump(json_data, dst, indent=2)


class DataProcessorMeta(Meta):

    @classmethod
    def _get_initial_data(cls):
        data = super()._get_initial_data()
        data['datasets'] = {}
        return data

    @classmethod
    def _to_json(cls, data):
        json_data = super()._to_json(data)
        json_data['datasets'] = {name: d.to_json() for name, d in data['datasets'].items()}
        return json_data

    @classmethod
    def _from_json(cls, json_data, *, parent_dir):
        data = super()._from_json(json_data, parent_dir=parent_dir)
        data['datasets'] = {name: DatasetSubMeta.from_json_data(data, parent_dir)
                            for name, data in json_data['datasets'].items()}
        return data

    def add_dataset(self, name, dataset):
        self._data['datasets'][name] = DatasetSubMeta.from_dataset(dataset, self._parent_dir)


class PreprocessorMeta(DataProcessorMeta):
    pass


class FeaturesMeta(DataProcessorMeta):
    pass


class CVMeta(Meta):

    def __init__(self, *args, folds_filename='folds.json', n_runs=3, n_folds=3, **kwargs):
        super().__init__(*args, **kwargs)
        self._data['folds_filename'] = folds_filename
        self._data['n_runs'] = n_runs
        self._data['n_folds'] = n_folds

    @property
    def folds_filename(self):
        return self._data['folds_filename']

    @property
    def n_runs(self):
        return self._data['n_runs']

    @property
    def n_folds(self):
        return self._data['n_folds']

    @classmethod
    def _get_initial_data(cls):
        data = super()._get_initial_data()
        # data['runs'] = []
        return data

    @classmethod
    def _to_json(cls, data):
        json_data = super()._to_json(data)
        for attr in ('folds_filename', 'n_runs', 'n_folds'):
            json_data[attr] = data[attr]
        # json_data['runs'] = {name: d.to_json() for name, d in data['datasets'].items()}
        return json_data

    @classmethod
    def _from_json(cls, json_data, *, parent_dir):
        data = super()._from_json(json_data, parent_dir=parent_dir)
        for attr in ('folds_filename', 'n_runs', 'n_folds'):
            data[attr] = json_data[attr]
        # data['runs'] = {name: DatasetSubMeta.from_json_data(data, parent_dir)
                            # for name, data in json_data['datasets'].items()}
        return data


class ModelMeta(Meta):

    @property
    def models(self):
        return self._data['models']

    @classmethod
    def _get_initial_data(cls):
        data = super()._get_initial_data()
        data['models'] = {}
        return data

    @classmethod
    def _to_json(cls, data):
        json_data = super()._to_json(data)
        json_data['models'] = {name: m.to_json() for name, m in data['models'].items()}
        return json_data

    @classmethod
    def _from_json(cls, json_data, *, parent_dir):
        data = super()._from_json(json_data, parent_dir=parent_dir)
        data['models'] = {name: ModelSubMeta.from_json_data(data, parent_dir)
                          for name, data in json_data['models'].items()}
        return data

    def add_model(self, name, model):
        self._data['models'][name] = ModelSubMeta.from_model(model, name, self._parent_dir)


class SubmissionMeta(Meta):
    pass


class SubMeta:
    pass


class PersistedObjectSubMeta(SubMeta):

    def __init__(self, data, parent_dir):
        self._data = data
        self._parent_dir = parent_dir

    @property
    def created_at(self):
        return self._data['created_at']

    @property
    def type(self):
        return self._data['type']

    @property
    def filename(self):
        return self._data['filename']

    @classmethod
    def from_json_data(cls, json_data, parent_dir):
        data = cls._from_json(json_data)
        obj = cls(data, parent_dir)
        return obj

    @classmethod
    def _to_json(cls, data):
        json_data = {
            'created_at': data['created_at'].isoformat(),
            'type': '{}.{}'.format(data['type'].__module__, data['type'].__name__),
            'filename': data['filename'],
            'params': data['params'],
        }
        return json_data

    @classmethod
    def _from_json(cls, json_data):
        data = {
            'created_at': dateutil.parser.parse(json_data['created_at']),
            'type': utils.import_class_by_path(json_data['type']),
            'filename': json_data['filename'],
            'params': json_data['params'],
        }
        return data

    def to_json(self):
        return self._to_json(self._data)

    def build_object(self):
        obj_type = self._data['type']
        obj_path = os.path.join(self._parent_dir, self._data['filename'])
        obj = obj_type.load(obj_path, **self._data['params'])
        return obj


class DatasetSubMeta(PersistedObjectSubMeta):

    @classmethod
    def from_dataset(cls, dataset, parent_dir):
        data = {
            'created_at': datetime.datetime.now(),
            'type': dataset.__class__,
            'filename': os.path.basename(dataset.filename),
            'params': dataset.params,
        }
        obj = cls(data, parent_dir)
        return obj


class ModelSubMeta(PersistedObjectSubMeta):

    @classmethod
    def from_model(cls, model, model_name, parent_dir):
        data = {
            'created_at': datetime.datetime.now(),
            'type': model.__class__,
            'filename': os.path.join(parent_dir, model_name),
            'params': {},
        }
        obj = cls(data, parent_dir)
        return obj
