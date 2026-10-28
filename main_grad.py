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
from sklearn.metrics import roc_auc_score, accuracy_score

random_seeds = 324221
SEED = 4534423
FC = 5
RANDOM_STATE = 432117

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)

random.seed(SEED)


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
        for df in (self.train_df, self.test_df):
            df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
            df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
            df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
            common = ["Mr", "Mrs", "Miss", "Master"]
            df["Title"] = df["Title"].where(df["Title"].isin(common), "Rare")
            df["Title"] = df["Title"].map({"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4})
            df["FamilyGroup"] = pd.cut(
                df["SibSp"] + df["Parch"] + 1,
                bins=[0, 1, 4, float("inf")],
                labels=[0, 1, 2]
            ).astype(int)

        age_med = self.train_df.groupby("Title")["Age"].median()
        for df in (self.train_df, self.test_df):
            df["Age"] = df["Age"].fillna(df["Title"].map(age_med))

        drop_cols = ["Name", "Ticket", "Cabin", "SibSp", "Parch", "Fare", "Embarked"]
        self.train_df.drop(columns=drop_cols, inplace=True)
        self.test_df.drop(columns=drop_cols, inplace=True)

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
        int_cols = df.select_dtypes(include=["integer"]).columns
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], downcast="integer")

        float_cols = df.select_dtypes(include=["floating"]).columns
        for col in float_cols:
            df[col] = pd.to_numeric(df[col], downcast="float")

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
                                        random_state=RANDOM_STATE)
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
    def training_phase(train, valid, nums, cats, target, seed=23143, rounds=150):
        train_dataset = lgb.Dataset(train[nums + cats],
                                    train[target],
                                    categorical_feature=cats)
        valid_dataset = lgb.Dataset(valid[nums + cats],
                                    valid[target],
                                    categorical_feature=cats)

        params = {
            "objective": "binary",
            "learning_rate": 0.01,
            "num_leaves": 7,  # или 6–8
            "max_depth": 4,
            "min_data_in_leaf": 25,
            "lambda_l2": 10,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "verbose": -1,
            "seed": seed
        }

        model = lgb.train(params=params,
                          train_set=train_dataset,
                          valid_sets=[valid_dataset], num_boost_round=rounds)

        esti = model.predict(valid[nums + cats])
        ms = roc_auc_score(valid[target],
                           esti)

        return model, ms


def main():
    global SEED, RANDOM_STATE

    base = LearningManager()
    random_SEED = [random.randint(1, 100000) for _ in range(10)]
    random_RANDOM_STATE = [random.randint(1, 100000) for _ in range(10)]
    res_pred = np.zeros(len(base.data_manager.test_df))

    for i in range(10):
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
                                          lm.data_manager.target, seed=SEED)

            test_pred += model.predict(lm.data_manager.test_df[feats])

            metrics.append(ms)

        test_pred /= FC
        print([np.round(ms, 6) for ms in metrics], np.mean(metrics))
        print()

        res_pred += test_pred

        SEED = random_SEED[i]
        RANDOM_STATE = random_RANDOM_STATE[i]

        np.random.seed(SEED)
        os.environ["PYTHONHASHSEED"] = str(SEED)

        #print(submission.head())

        # print("train\n", train_df_down.head())
        # print("test\n", test_df_down.head())

    res_pred /= 10

    submission = pd.DataFrame({
        "PassengerId": base.data_manager.test_df["PassengerId"],
        "Survived": (res_pred > 0.5).astype(int),
    })
    submission.to_csv('data/gender_submission.csv',
                      index=False)


if __name__ == "__main__":
    main()
