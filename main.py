import random
import os
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz

from tqdm import tqdm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score

SEED = 67691488
FC = 5
RANDOM_STATE = 2138

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


class DataManager:
    def __init__(self):
        self.train_df: pd.DataFrame = pd.read_csv("data/train.csv")
        self.test_df: pd.DataFrame = pd.read_csv("data/test.csv")

        self.target = "Survived"
        self.inds = []

        self.preprocess_df()

        self.nums = ([self.train_df[i].name for i in self.train_df.select_dtypes(include=["integer"]).columns if
                      self.train_df[i].name != "PassengerId" and self.train_df[i].name != "Survived"]
                     + [self.train_df[i].name for i in self.train_df.select_dtypes(include=["floating"]).columns if
                        self.train_df[i].name != "PassengerId" and self.train_df[i].name != "Survived"])
        self.cats = [c for c in self.train_df.columns if c not in self.nums and c != self.target and c != "PassengerId"]

    def preprocess_df(self):
        self.train_df["Sex"] = self.train_df["Sex"].map({"male": 1, "female": 0})
        self.test_df["Sex"] = self.test_df["Sex"].map({"male": 1, "female": 0})

        self.train_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1, inplace=True)
        self.test_df.drop(["Name", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], axis=1, inplace=True)

        self.train_df = DataManager.downcast_df(self.train_df)
        self.test_df = DataManager.downcast_df(self.test_df)

        self.cross_validation()

    def cross_validation(self):
        kf = KFold(n_splits=FC,
                   shuffle=True,
                   random_state=RANDOM_STATE)
        for (train_index, valid_index) in kf.split(self.train_df):
            self.inds += [[train_index, valid_index]]

    @staticmethod
    def downcast_df(df: pd.DataFrame) -> pd.DataFrame:
        start = df.memory_usage().sum() / 1024 ** 2

        int_cols = df.select_dtypes(include=["integer"]).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        float_cols = df.select_dtypes(include=["floating"]).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast="float")

        end = df.memory_usage().sum() / 1024 ** 2

        print('memory usage before optimization is: {:.2f} MB'.format(start))
        print('memory usage after optimization is: {:.2f} MB'.format(end))
        print('decreased by {:.1f}%'.format(100 * (start - end) / start))
        print()

        return df


class LearningManager:
    def __init__(self):
        self.data_manager: DataManager = DataManager()

        self.destroy_spy()

    def destroy_spy(self):
        self.data_manager.train_df["av"] = 0
        self.data_manager.test_df["av"] = 1

        fr = pd.concat([self.data_manager.train_df, self.data_manager.test_df],
                       axis=0)
        fr.reset_index(drop=True,
                       inplace=True)

        train, valid = train_test_split(fr,
                                        train_size=.80,
                                        test_size=.20,
                                        random_state=1723)
        train.reset_index(drop=True,
                          inplace=True)
        valid.reset_index(drop=True,
                          inplace=True)

        cs = {}
        tr = True

        while tr:
            n = [c for c in self.data_manager.nums if c not in cs]
            c = [c for c in self.data_manager.cats if c not in cs]

            if not (n or c):
                break

            model, metrics = LearningManager.training_phase(train, valid, n, c, "av")

            if metrics > .63:

                ind = np.argmax(model.feature_importance("gain"))
                feats_space = n + c

                print(feats_space[ind],
                      metrics)

                cs[feats_space[ind]] = metrics

            else:
                tr = False

        self.data_manager.nums = [c for c in self.data_manager.nums if c not in cs]
        self.data_manager.cats = [c for c in self.data_manager.cats if c not in cs]
        for key in cs.keys():
            self.data_manager.train_df.drop(key,
                                            axis=1,
                                            inplace=True)
            self.data_manager.test_df.drop(key,
                                           axis=1,
                                           inplace=True)

        self.data_manager.train_df.drop("av",
                                        axis=1,
                                        inplace=True)
        self.data_manager.test_df.drop("av",
                                       axis=1,
                                       inplace=True)

    @staticmethod
    def training_phase(train, valid, nums, cats, target):
        train_dataset = lgb.Dataset(train[nums + cats],
                                    train[target],
                                    categorical_feature=cats)
        valid_dataset = lgb.Dataset(valid[nums + cats],
                                    valid[target],
                                    categorical_feature=cats)

        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "num_boost_round": 1000,
            "early_stopping_round": 100,
            "max_depth": -1,
            "bagging_fraction": .9,
            "feature_fraction": 1,
            "verbose": 0,
            "nthreads": 4,
            "random_state": 1723,
            "bagging_freq": 1,
            "learning_rate": 0.05,
            "num_leaves": 8,
            "min_data_in_leaf": 15,
            "metric": "auc",
            "min_gain_to_split": 0
        }

        model = lgb.train(params=params,
                          train_set=train_dataset,
                          valid_sets=[valid_dataset])

        esti = model.predict(valid[nums + cats])
        ms = roc_auc_score(valid[target],
                           esti)

        return model, ms


def main():
    lm: LearningManager = LearningManager()

    test_pred = np.zeros(len(lm.data_manager.test_df))
    feats = lm.data_manager.nums + lm.data_manager.cats

    metrics = []

    for fold in tqdm(range(FC)):

        train = lm.data_manager.train_df.iloc[lm.data_manager.inds[fold][0]]
        train.reset_index(drop=True,
                          inplace=True)

        valid = lm.data_manager.train_df.iloc[lm.data_manager.inds[fold][1]]
        valid.reset_index(drop=True,
                          inplace=True)

        model, ms = lm.training_phase(train, valid,
                                      lm.data_manager.nums, lm.data_manager.cats,
                                      lm.data_manager.target)

        test_pred += model.predict(lm.data_manager.test_df[feats])

        metrics.append(ms)

    test_pred /= FC
    print([np.round(ms, 6) for ms in metrics], np.mean(metrics))

    submission = pd.DataFrame({
        "PassengerId": lm.data_manager.test_df["PassengerId"],
        "Survived": (test_pred > 0.5).astype(int),
    })
    submission.to_csv('data/gender_submission.csv',
                      index=False)

    #print(submission.head())

    # print("train\n", train_df_down.head())
    # print("test\n", test_df_down.head())


if __name__ == "__main__":
    main()
