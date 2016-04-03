import json
import os

import numpy as np
import sklearn.cross_validation


class CrossValidator:

    def __init__(self, dataset_name, target_col, model, metric):
        self.dataset_name = dataset_name
        self.target_col = target_col
        self.model = model
        self.metric = metric

    def run(self, dataframe, meta):
        """
        Fit and score the model on all folds.
        """

        folds = self._get_or_create_folds(dataframe, meta)

        scores = []

        for per_run_folds in folds:
            for train_idx, test_idx in per_run_folds:
                train_df = dataframe.ix[train_idx]
                test_df = dataframe.ix[test_idx]

                self.model.fit(self.target_col, train_df, test_df)

                y_true = test_df[self.target_col]
                y_pred = self.model.predict(test_df.drop(self.target_col, axis=1))

                score = self.metric(y_true, y_pred)
                print(score)

                scores.append(score)

        print('Mean: {:.5f}'.format(np.mean(scores)))
        print('Std:  {:.5f}'.format(np.std(scores)))

    def _get_or_create_folds(self, dataframe, meta):
        """Load folds from file or generate new ones if file doesn't exist."""

        folds_path = os.path.join(meta.directory, meta.folds_filename)

        if os.path.exists(folds_path):
            folds = self._load_folds(folds_path)
        else:
            folds = self._gen_folds(dataframe, meta.n_runs, meta.n_folds)
            self._save_folds(folds, folds_path)

        return folds

    def _load_folds(self, path):
        """Save folds to file."""
        with open(path) as folds_file:
            folds = json.load(folds_file)
        return folds

    def _save_folds(self, folds, path):
        """Load folds from file."""
        with open(path, 'w') as folds_file:
            json.dump(folds, folds_file)

    def _gen_folds(self, dataframe, n_runs, n_folds):
        """
        Create new folds.

        Args:
            dataframe: pandas.DataFrame used to derive indexes.
            n_runs: Number of runs.
            n_folds: Number of folds per run.

        Returns:
            A list containing folds for each run.
            Folds for a run are a list of (train, test) index pairs.
        """

        assert dataframe.index.is_integer(), 'Only integer indexes are expected'

        runs = []

        for _ in range(n_runs):
            folds = []
            fold_idxs = sklearn.cross_validation.KFold(dataframe.index.size,
                                                       n_folds=n_folds,
                                                       shuffle=True)
            for train_idx, test_idx in fold_idxs:
                # Convert indexes to native types for smooth serialization.
                train_df_ix = dataframe.index[train_idx].map(int).tolist()
                test_df_ix = dataframe.index[test_idx].map(int).tolist()
                pair = (train_df_ix, test_df_ix)
                folds.append(pair)
            runs.append(folds)

        return runs
