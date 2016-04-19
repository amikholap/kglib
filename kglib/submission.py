import os

import pandas as pd


class SubmissionMaker:

    def __init__(self, submission_id, model_id, dataset_name):
        self.submission_id = submission_id
        self.model_id = model_id
        self.dataset_name = dataset_name

    def run(self, model, dataframe, result_col_name, meta):
        """Prepare a submission and save it to a CSV file."""
        predictions = model.predict(dataframe)
        predictions = pd.DataFrame(predictions, columns=[result_col_name])
        path = os.path.join(meta.directory, self.submission_id)
        predictions.to_csv(path)
