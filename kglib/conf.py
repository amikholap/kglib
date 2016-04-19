import inspect
import logging.config
import os

from .meta import Meta, CVMeta, FeaturesMeta, ModelMeta, PreprocessorMeta
from .utils import ensure_dir_exists


class BaseConfig:
    """A class containing a complete description of a project."""

    def __init__(self, data_dir='data', assets_dir='assets'):

        # Project root directory.
        self.root_dir = self._get_root_dir()

        # Directory containing source data.
        self.data_dir = os.path.join(self.root_dir, data_dir)

        # Directory to store features, models, etc.
        self.assets_dir = os.path.join(self.root_dir, assets_dir)
        ensure_dir_exists(self.assets_dir)

        preprocessed_dir = os.path.join(self.assets_dir, 'preprocessed')
        ensure_dir_exists(preprocessed_dir)
        self.preprocessed_meta = PreprocessorMeta(os.path.join(preprocessed_dir, 'meta.json'))

        features_dir = os.path.join(self.assets_dir, 'features')
        ensure_dir_exists(features_dir)
        self.features_meta = FeaturesMeta(os.path.join(features_dir, 'meta.json'))

        cv_dir = os.path.join(self.assets_dir, 'cv')
        ensure_dir_exists(cv_dir)
        self.cv_meta = CVMeta(os.path.join(cv_dir, 'meta.json'))

        model_dir = os.path.join(self.assets_dir, 'models')
        ensure_dir_exists(model_dir)
        self.model_meta = ModelMeta(os.path.join(model_dir, 'meta.json'))

        submission_dir = os.path.join(self.assets_dir, 'submissions')
        ensure_dir_exists(submission_dir)
        self.submission_meta = Meta(os.path.join(submission_dir, 'meta.json'))

        self._logging_config = {
            'version': 1,

            # Project loggers are created at import time
            # while this configuration is applied after.
            # We don't want to disable that loggers.
            'disable_existing_loggers': False,

            'formatters': {
                'simple': {
                    'format': '%(levelname)s %(asctime)s %(message)s'
                },
            },

            'handlers': {
                'console': {
                    'level': 'INFO',
                    'class': 'logging.StreamHandler',
                    'formatter': 'simple'
                },
            },

            'root': {
                'level': 'INFO',
                'handlers': ['console'],
            },
        }

    def _get_root_dir(self):
        """
        Returns path to project's root directory.

        The path is inferred from config instance's file.
        """
        cls = self.__class__
        cls_file = inspect.getfile(cls)
        src_dir = os.path.dirname(cls_file)
        root_dir = os.path.dirname(src_dir)
        return root_dir

    def configure_logging(self):
        """Configure system-wide loggers."""
        logging.config.dictConfig(self._logging_config)
