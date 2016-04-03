import logging
import time

from .datasets import PandasCsvDataset


LOGGER = logging.getLogger(__name__)


class DataProcessor:

    dataset_type = PandasCsvDataset

    def __init__(self, inputs, outputs, **kwargs):
        # Convert single input/output to a list.
        inputs = [inputs] if isinstance(inputs, str) else list(inputs)
        outputs = [outputs] if isinstance(outputs, str) else list(outputs)

        self.inputs = inputs
        self.outputs = outputs

        self.kwargs = kwargs

    def __hash__(self):
        str_id = '{}:{}:{}'.format(self.__class__.__name__,
                                   ','.join(self.inputs),
                                   ','.join(self.outputs))
        return hash(str_id)

    def __str__(self):
        string = '{} {} -> {}'.format(
            self.__class__.__name__,
            ','.join(self.inputs),
            ','.join(self.outputs),
        )
        return string

    def process(self, dataframes):

        self._log_processing_start(LOGGER)

        start_time = time.time()

        assert len(self.inputs) == len(dataframes)
        output_dataframes = self._do_process(dataframes, **self.kwargs)
        assert len(self.outputs) == len(output_dataframes)

        end_time = time.time()

        LOGGER.info('Elapsed time: %.2f seconds', end_time - start_time)

        return output_dataframes

    def _log_processing_start(self, logger):
        log_fmt = 'Applying %s'
        log_args = [str(self)]
        if self.kwargs:
            log_fmt += '\nArguments:'
            for key, val in self.kwargs.items():
                log_fmt += '\n%s=%s'
                log_args.extend([key, val])
        logger.info(log_fmt, *log_args)

    def _do_process(self, dataframes, **kwargs):
        raise NotImplementedError

    def get_dataset_params(self, dataframes):
        """Parameters for each output dataset."""
        assert len(dataframes) == len(self.outputs)
        params = [self._get_generic_dataset_params(df) for df in dataframes]
        return params

    def _get_generic_dataset_params(self, dataframe):
        """Common dataframe parameters derived from dataframe characteristics."""

        params = {
            'dataframe': dataframe,
            'read_csv_params': {},
            'to_csv_params': {},
        }

        # Remember index column for natural index.
        # Don't write index column for surrogate index.
        if dataframe.index.name:
            params['read_csv_params']['index_col'] = dataframe.index.name
        else:
            params['to_csv_params']['index'] = False

        return params
