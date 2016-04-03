from .data_processor import DataProcessor


class FeatureExtractor(DataProcessor):

    def _do_process(self, dataframes, **kwargs):
        raise NotImplementedError


class JoinFeatureExtractor(FeatureExtractor):

    def _do_process(self, dataframes):
        joined = dataframes[0].copy()
        for dataframe in dataframes[1:]:
            joined = joined.join(dataframe.copy())
        return joined,
