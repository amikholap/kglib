import logging
import os

from .cross_val import CrossValidator
from .feature_extractors import FeatureExtractor
from .model_maker import ModelMaker
from .preprocessors import Preprocessor
from .submission import SubmissionMaker


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
            elif isinstance(action, ModelMaker):
                self.run_model_maker(action, self.config.model_meta, datasets)
            elif isinstance(action, SubmissionMaker):
                model = self._load_model(action.model_id, self.config.model_meta)
                self.run_submission_maker(action, model, self.config.submission_meta, datasets)
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
                dataset = meta.datasets[name].build_object()
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

    def run_model_maker(self, model_maker, meta, datasets):
        dataframe = datasets[model_maker.dataset_name].dataframe
        model_maker.run(dataframe, meta)
        meta.add_model(model_maker.model_id, model_maker.model)
        meta.save()

    def run_submission_maker(self, submission_maker, model, meta, datasets):
        dataframe = datasets[submission_maker.dataset_name].dataframe
        submission_maker.run(model, dataframe, 'relevance', meta)
        meta.save()

    def _load_model(self, model_id, meta):
        model = meta.models[model_id].build_object()
        return model
