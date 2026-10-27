import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import dataset_manager as dm
import model_utils as utils

class LGDManager:
    def __init__(self, train_manager, test_manager):
        self.train_valid_man = train_manager
        self.test_man = test_manager
        self.train_valid_man.drop_columns(
            ["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
        )
        self.test_man.drop_columns(
            ["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
        )
        self.train_valid_man.train_valid_split("Survived")

        self.train_lgb = self._convert_df_to_lbm_dataset(self.train_valid_man.train_df,
                                                         self.train_valid_man.train_labels)
        self.valid_lgb = self._convert_df_to_lbm_dataset(self.train_valid_man.valid_df,
                                                         self.train_valid_man.valid_labels)

    @staticmethod
    def _convert_df_to_lbm_dataset(df, label):
        return lgb.Dataset(data=df, label=label)


params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "num_iterations": 10,
    "learning_rate": 0.05,
    "num_leaves": 4,
    "num_threads": 4,
}

if __name__ == "__main__":
    train_manager = dm.TrainDatasetManager("train.csv")
    test_manager = dm.TestDatasetManager("test.csv", "PassengerId")
    l_manager = LGDManager(train_manager, test_manager)

    model = lgb.train(
        params=params,
        train_set=l_manager.train_lgb,
        valid_sets=[l_manager.valid_lgb],
    )

    esti = model.predict(train_manager.valid_df)
    ms = roc_auc_score(train_manager.valid_labels, esti)
    print(ms)

    esti = model.predict(test_manager.get_df())
    test_manager.update_predictions(
        prediction_name="Survived",
        predictions=utils.binary_postprocess(esti),
    )
    test_manager.form_submit("submit.csv")