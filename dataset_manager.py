import pandas as pd
from sklearn.model_selection import train_test_split, KFold


class DatasetManager:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name, encoding="utf-8")
        self._downcast_for_df()

    def _downcast_for_df(self):
        int_cols = self.df.select_dtypes(include=["integer"]).columns
        for col in int_cols:
            self.df[col] = pd.to_numeric(self.df[col], downcast="integer")
        float_cols = self.df.select_dtypes(include=["float"]).columns
        for col in float_cols:
            self.df[col] = pd.to_numeric(self.df[col], downcast="float")
        object_cols = self.df.select_dtypes(include=["object"]).columns
        for col in object_cols:
            self.df[col] = self.df[col].astype("category")

    def drop_columns(self, list_of_columns):
        self.df.drop(list_of_columns, axis=1, inplace=True)

    def get_df(self):
        return self.df


class TrainDatasetManager(DatasetManager):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.train_df = None
        self.valid_df = None
        self.train_labels = None
        self.valid_labels = None
        self.cross_valid_folds = []

    def train_valid_split(self, target_label):
        self.train_df, self.valid_df, self.train_labels, self.valid_labels = train_test_split(
            self.df.drop([target_label], axis=1), self.df[target_label], test_size=0.2, random_state=2443)

    def prepare_folds(self, number_of_folds, target_label):
        kf = KFold(n_splits=number_of_folds, shuffle=True, random_state=64774)
        for train_index, valid_index in kf.split(self.get_df().drop([target_label], axis=1),
                                                 self.get_df()[target_label]):
            self.cross_valid_folds.append((train_index, valid_index))


class TestDatasetManager(DatasetManager):
    def __init__(self, file_name, id_name):
        super().__init__(file_name)
        self.id_name = id_name

    def update_predictions(self, prediction_name, predictions):
        self.df = pd.DataFrame(self.df[self.id_name])
        self.df[prediction_name] = predictions

    def form_submit(self, save_to_file_name):
        self.df.to_csv(save_to_file_name, index=False, encoding="utf-8", mode="w")

    def get_df(self):
        return super().get_df().drop([self.id_name], axis=1)
