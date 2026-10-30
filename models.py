import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss


class EnsembleBinaryClassifier:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_iterations": 1000,
        "num_leaves": 12,
        "max_depth": 4,
        "min_data_in_leaf": 12,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "num_threads": 1,
        "verbose": -1,
        "max_cat_to_onehot": 8
    }

    def __init__(self, target_name):
        self.models = []
        self.target_name = target_name

    def _train_model(self, train_fold, valid_fold):
        model_to_fit = lgb.train(
            params=self.params,
            train_set=train_fold,
            valid_sets=[valid_fold],
            callbacks=[
                lgb.early_stopping(100)
            ]
        )
        self.models.append(model_to_fit)

    def _unpack_fold(self, train_id, valid_id, train_df):
        train = lgb.Dataset(
            train_df.drop([self.target_name], axis=1).iloc[train_id],
            train_df[self.target_name].iloc[train_id]
        )

        valid = lgb.Dataset(
            train_df.drop([self.target_name], axis=1).iloc[valid_id],
            train_df[self.target_name].iloc[valid_id]
        )
        return train, valid

    def train_ensemble(self, train_man):
        df = train_man.get_df()
        X = df.drop([self.target_name], axis=1)
        y = df[self.target_name]
        oof = np.zeros(len(df))

        for train_id, valid_id in train_man.cross_valid_folds:
            self._train_model(*self._unpack_fold(train_id, valid_id, df))
            oof[valid_id] = self.models[-1].predict(X.iloc[valid_id])

        print("OOF accuracy:", accuracy_score(y, oof >= 0.5))
        print("OOF logloss: ", log_loss(y, oof))
        return oof

    def predict(self, df):
        predictions = np.zeros(len(df))
        for model in self.models:
            predictions += model.predict(df)
        predictions /= len(self.models)
        return predictions

    def roc_auc_score_calc(self, target_col, validation_df):
        return roc_auc_score(target_col, self.predict(validation_df))

    @staticmethod
    def binary_postprocess(predictions):
        return [int(i >= 0.5) for i in predictions]
