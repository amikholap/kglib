import logging
import re

import nltk

from .data_processor import DataProcessor


LOGGER = logging.getLogger(__name__)


class Preprocessor(DataProcessor):

    def _do_process(self, dataframes, **kwargs):
        raise NotImplementedError


class ExtractColumnPreprocessor(Preprocessor):

    def _do_process(self, dataframes, *, col_name):
        dataframe, = dataframes
        column = dataframe[col_name].to_frame()
        return column,


class FillNanPreprocessor(Preprocessor):

    def _do_process(self, dataframes, *, columns, fill_value):
        output_dataframes = []

        for input_df in dataframes:
            output_df = input_df.copy()
            for col in columns:
                output_df[col].fillna(fill_value, inplace=True)
            output_dataframes.append(output_df)

        return output_dataframes


class StringReplacementPreprocessor(Preprocessor):

    def _do_process(self, dataframes, *, columns, substitutions):
        output_dataframes = []

        for input_df in dataframes:
            output_df = input_df.copy()
            for col in columns:
                for pattern, repl in substitutions:
                    output_df[col] = output_df[col].apply(
                        lambda x, p=pattern, r=repl: re.sub(p, r, x)
                    )
            output_dataframes.append(output_df)

        return output_dataframes


class StemmerPreprocessor(Preprocessor):

    def _do_process(self, dataframes, *, columns):
        output_dataframes = []

        for input_df in dataframes:
            output_df = input_df.copy()

            stemmer = nltk.PorterStemmer()

            for col in columns:
                data = output_df[col]
                data = data.apply(lambda text: text.lower())
                data = data.apply(nltk.word_tokenize)
                data = data.apply(lambda tokens: ' '.join([stemmer.stem(t) for t in tokens]))
                data.replace('', ' ', inplace=True)  # Save & load of '' results to NaN.
                output_df[col] = data

            output_dataframes.append(output_df)

        return output_dataframes
