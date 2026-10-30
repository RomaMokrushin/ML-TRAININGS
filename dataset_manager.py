import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import lightgbm as lgb

SEED = 1245134


class DatasetManager:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name, encoding="utf-8")
        print(self.df.dtypes)

    def downcast_for_df(self):
        int_cols = self.df.select_dtypes(include=["integer"]).columns
        for col in int_cols:
            self.df[col] = pd.to_numeric(self.df[col], downcast="integer")
        float_cols = self.df.select_dtypes(include=["float"]).columns
        for col in float_cols:
            self.df[col] = pd.to_numeric(self.df[col], downcast="float")

    def drop_columns(self, list_of_columns):
        self.df.drop(list_of_columns, axis=1, inplace=True)

    def get_df(self):
        return self.df

    def keep_columns(self, columns):
        self.df = self.df[columns]

    def column_element_apply_callback(self, column_label, callback, new_label=None):
        if not new_label:
            new_label = column_label
        self.df[new_label] = self.df[column_label].apply(lambda x: callback(x))

    def apply_callback(self, column_label : str | list, callback):
        callback(self.df[column_label])

    def column_astype(self, column_label, dtype, new_label=None):
        if not new_label:
            new_label = column_label
        self.df[new_label] = self.df[column_label].astype(dtype)


class TrainDatasetManager(DatasetManager):
    def __init__(self, file_name):
        super().__init__(file_name)
        self.cross_valid_folds = []

    def prepare_folds(self, number_of_folds, target_label, seed=SEED):
        self.cross_valid_folds = []
        skf = StratifiedKFold(n_splits=number_of_folds, shuffle=True, random_state=seed)
        y = self.get_df()[target_label]
        for train_index, valid_index in skf.split(self.get_df(), y):
            self.cross_valid_folds.append((train_index, valid_index))


class TestDatasetManager(DatasetManager):
    def __init__(self, file_name, id_name):
        super().__init__(file_name)
        self.id_name = id_name

    def update_predictions(self, target_label, predictions):
        self.df = pd.DataFrame(self.df[self.id_name])
        self.df[target_label] = predictions

    def form_submit(self, save_to_file_name):
        self.df.to_csv(save_to_file_name, index=False, encoding="utf-8")

    def get_df(self):
        return super().get_df().drop([self.id_name], axis=1)

    def keep_columns(self, columns):
        super().keep_columns(columns + [self.id_name])


class DatasetsControl:
    def __init__(
            self,
            train_manager,
            test_manager,
            target_label
    ):
        self.target_label: str = target_label
        self.train_manager: TrainDatasetManager = train_manager
        self.test_manager: TestDatasetManager = test_manager

    def clean_dss(self, columns: list):
        if not columns:
            return
        self.train_manager.keep_columns(columns)
        test_c = [c for c in columns if c != self.target_label]
        self.test_manager.keep_columns(test_c)

    def one_cat(self, column_label):
        cats = pd.concat([self.train_manager.get_df()[column_label],
                            self.test_manager.get_df()[column_label]])
        cats = cats.dropna().unique()
        cats = pd.CategoricalDtype(categories=cats)

        self.train_manager.column_astype(column_label, cats)
        self.test_manager.column_astype(column_label, cats)

    def elements_callback(self, column_label, callbacks : list, new_label=None):
        for callback in callbacks:
            self.train_manager.column_element_apply_callback(column_label, callback, new_label)
            self.test_manager.column_element_apply_callback(column_label, callback, new_label)

    def form_lgb_datasets(self, test_size):
        df = self.train_manager.get_df()
        train_df, valid_df, train_labels, valid_labels = train_test_split(
            df.drop([self.target_label], axis=1),
            df[self.target_label],
            test_size=test_size,
            random_state=SEED
        )
        train_lgb = lgb.Dataset(train_df,
                                train_labels)
        valid_lgb = lgb.Dataset(valid_df,
                                valid_labels)
        return train_lgb, valid_lgb
