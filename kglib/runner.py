import logging
import os

from .cross_val import CrossValidator
from .preprocessors import Preprocessor
from .feature_extractors import FeatureExtractor


LOGGER = logging.getLogger(__name__)


class Runner:

    def __init__(self, config):
        self.config = config

    def run(self):
        self.config.configure_logging()

        datasets = self.config.sources.copy()

        for action in self.config.actions:
            if isinstance(action, Preprocessor):
                self.run_data_processor(action, self.config.preprocessed_meta, datasets)
            elif isinstance(action, FeatureExtractor):
                self.run_data_processor(action, self.config.features_meta, datasets)
            elif isinstance(action, CrossValidator):
                self.run_cv(action, self.config.cv_meta, datasets)
            else:
                raise RuntimeError('Unknown action "{}"'.format(str(action)))

    def run_data_processor(self, data_processor, meta, datasets):
        """Apply a preprocessor or a feature extractor."""

        cache_available = True
        output_paths = []

        for name in data_processor.outputs:
            if name in meta.datasets:
                output_path = meta.datasets[name].filename
            else:
                output_path = os.path.join(meta.directory, name)
                cache_available = False
            output_paths.append(output_path)

        if cache_available:
            LOGGER.info('Using cache for %s', data_processor)
        else:
            input_dataframes = [datasets[name].dataframe for name in data_processor.inputs]
            output_dataframes = data_processor.process(input_dataframes)
            params = data_processor.get_dataset_params(output_dataframes)

        for i, name in enumerate(data_processor.outputs):
            if cache_available:
                dataset = meta.datasets[name].build_dataset()
            else:
                dataset = data_processor.dataset_type(output_paths[i], **params[i])
                dataset.save()
                meta.add_dataset(name, dataset)
                meta.save()

            datasets[name] = dataset

    def run_cv(self, cross_validator, meta, datasets):
        dataframe = datasets[cross_validator.dataset_name].dataframe
        cross_validator.run(dataframe, meta)
        meta.save()
