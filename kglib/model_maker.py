import os


class ModelMaker:

    def __init__(self, model_id, model, dataset_name, target_col, metric):
        self.model_id = model_id
        self.model = model
        self.dataset_name = dataset_name
        self.target_col = target_col
        self.metric = metric

    def run(self, dataframe, meta):
        """Fit the model and save it."""
        self.model.fit(self.target_col, dataframe)
        path = os.path.join(meta.directory, self.model_id)
        self.model.save(path)
