import pandas as pd
import sklearn as sk


class DatasetManager:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name, encoding="utf-8")
        self.train_df = None
        self.valid_df = None
        self.train_labels = None
        self.valid_labels = None
        self._downcast_for_df()
        self._train_valid_split()

    def _downcast_for_df(self):
        int_cols = self.df.select_dtypes(include=["integer"]).columns
        for col in int_cols:
            self.df[col] = pd.to_numeric(self.df[col], downcast="integer")
        float_cols = self.df.select_dtypes(include=["float"]).columns
        for col in float_cols:
            self.df[col] = pd.to_numeric(self.df[col], downcast="float")

    def _train_valid_split(self):
        self.train_df, self.valid_df, self.train_labels, self.valid_labels = sk.train_test_split(
            self.df.drop(["survival"]), self.df.drop(["survival"]), test_size=0.2, random_state=666437)

class DatasetFormer:
    def __init__(self, file_name, df):
        self.file_name = file_name
        self.df = df

    def _form_submit(self):
        self.df.to_csv(self.file_name, index=False)
