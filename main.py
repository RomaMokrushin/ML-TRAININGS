import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import dataset_manager as dm
import model_utils as utils


class LGDManager:
    def __init__(self, train_manager, test_manager):
        self.train_valid_man = train_manager
        self.test_man = test_manager
        self.train_valid_man.drop_columns(
            ["PassengerId", "Name", "Parch", "Ticket", "Cabin", "Embarked"]
        )
        self.test_man.drop_columns(
            ["Name", "Parch", "Ticket", "Cabin", "Embarked"]
        )
        self.train_valid_man.train_valid_split("Survived")

        self.train_lgb = utils.df_to_lbm(self.train_valid_man.train_df,
                                         self.train_valid_man.train_labels)
        self.valid_lgb = utils.df_to_lbm(self.train_valid_man.valid_df,
                                         self.train_valid_man.valid_labels)


params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_iterations": 40,
    "learning_rate": 0.02,
    "num_leaves": 12,
    "num_threads": 1,
    "verbose": -1
}


class EnsembleClassifier:
    def __init__(self, number_of_models, target_name):
        self.number_of_models = number_of_models
        self.models = []
        self.target_name = target_name

    def _train_model(self, train_fold, valid_fold):
        model_to_fit = lgb.train(
            params=params,
            train_set=train_fold,
            valid_sets=[valid_fold],
        )
        self.models.append(model_to_fit)

    def _unpack_fold(self, train_id, valid_id, train_df):
        train = utils.df_to_lbm(
            train_df.drop([self.target_name], axis=1).iloc[train_id],
            train_df[self.target_name].iloc[train_id]
        )

        valid = utils.df_to_lbm(
            train_df.drop([self.target_name], axis=1).iloc[valid_id],
            train_df[self.target_name].iloc[valid_id]
        )
        return train, valid

    def _unpack_dataframe_valid(self, valid_id, train_df):
        return (train_df.drop([self.target_name], axis=1).iloc[valid_id],
                train_df[self.target_name].iloc[valid_id])

    def train_ensemble(self, train_man):
        roc_auc_per_model = []
        for number in range(self.number_of_models):
            fold = self._unpack_fold(*train_man.cross_valid_folds[number], train_man.get_df())
            self._train_model(*fold)
            valid_df_x, valid_df_y = self._unpack_dataframe_valid(train_man.cross_valid_folds[number][1],
                                                                  train_man.get_df())
            esti = self.models[number].predict(valid_df_x)
            roc_auc_per_model.append(roc_auc_score(valid_df_y, esti))
        return roc_auc_per_model

    def predict(self, df):
        predictions = pd.DataFrame({"0": np.zeros(len(df))})
        for model in self.models:
            predictions["0"] += model.predict(df)
        predictions["0"] /= self.number_of_models
        return predictions["0"].to_list()

    def roc_auc_score_calc(self, target_col, validation_df):
        return roc_auc_score(target_col, self.predict(validation_df))


if __name__ == "__main__":
    train_manager = dm.TrainDatasetManager("train.csv")
    test_manager = dm.TestDatasetManager("test.csv", "PassengerId")
    l_manager = LGDManager(train_manager, test_manager)

    model = EnsembleClassifier(number_of_models=5, target_name="Survived")
    train_manager.prepare_folds(number_of_folds=5, target_label="Survived")
    print(model.train_ensemble(train_man=train_manager))
    test_manager.update_predictions(
        prediction_name="Survived",
        predictions=utils.binary_postprocess(model.predict(test_manager.get_df())),
    )
    test_manager.form_submit("submit.csv")

