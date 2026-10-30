import pandas as pd

import models
import dataset_manager as dm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

ENSEMBLE_MODELS_NUMBER = 5
TARGET_LABEL = "Survived"
ID_NAME = "PassengerId"
WORK_COLUMNS = [TARGET_LABEL, "Pclass", "Sex", "Age",
                "Fare", "Name", "Ticket"]


class TitanicFeatureEngineering:
    def __init__(self, df_manager):
        self.manager: dm.DatasetsControl = df_manager
        self.cooking()

    def cooking(self):
        m = self.manager
        m.one_cat("Sex")
        # m.one_cat("Embarked")
        # m.elements_callback("Cabin", [self.get_floor])
        # m.one_cat("Cabin")

        m.elements_callback("Name", [self.get_surname], "Family")
        self.get_family_members()
        self.get_fare_per_person()
        m.elements_callback("Name", [self.get_rank])
        m.one_cat("Name")

        m.train_manager.downcast_for_df()
        m.test_manager.downcast_for_df()

    def get_family_members(self):
        m = self.manager
        tr_df = m.train_manager.df["Family"]
        ts_df = m.test_manager.df["Family"]
        df = pd.concat([tr_df, ts_df]).value_counts()
        m.train_manager.df["Family"] = tr_df.map(df)
        m.test_manager.df["Family"] = ts_df.map(df)

    def get_fare_per_person(self):
        m = self.manager
        tr_ticket = m.train_manager.df["Ticket"]
        ts_ticket = m.test_manager.df["Ticket"]
        counts = pd.concat([tr_ticket, ts_ticket]).value_counts()

        m.train_manager.df["TicketFreq"] = tr_ticket.map(counts).astype("int16")
        m.test_manager.df["TicketFreq"] = ts_ticket.map(counts).astype("int16")

        m.train_manager.df["FarePerPerson"] = m.train_manager.df["Fare"] / m.train_manager.df["TicketFreq"]
        m.test_manager.df["FarePerPerson"] = m.test_manager.df["Fare"] / m.test_manager.df["TicketFreq"]

        m.train_manager.drop_columns(["Ticket"])
        m.test_manager.drop_columns(["Ticket"])
        m.train_manager.drop_columns(["TicketFreq"])
        m.test_manager.drop_columns(["TicketFreq"])

    @staticmethod
    def get_rank(name: str):
        t = name.split(',')[1].split('.')[0].strip()
        t = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}.get(t, t)
        return t if t in ("Mr", "Mrs", "Miss", "Master") else "U"

    @staticmethod
    def get_surname(name: str):
        t = name.split(',')[0].strip()
        return t

    @staticmethod
    def get_floor(cabin: str):
        return cabin[0] if isinstance(cabin, str) else "U"


if __name__ == "__main__":
    # init
    train_manager = dm.TrainDatasetManager("train.csv")
    test_manager = dm.TestDatasetManager("test.csv", ID_NAME)
    datasets_control = dm.DatasetsControl(
        train_manager=train_manager,
        test_manager=test_manager,
        target_label=TARGET_LABEL
    )
    # prepare_for_dss
    datasets_control.clean_dss(WORK_COLUMNS)
    train_manager.prepare_folds(ENSEMBLE_MODELS_NUMBER, TARGET_LABEL)
    TitanicFeatureEngineering(datasets_control)
    # model init
    model = models.EnsembleBinaryClassifier(
        target_name=TARGET_LABEL
    )
    model.train_ensemble(train_manager)
    # test predict
    test_manager.update_predictions(
        target_label=TARGET_LABEL,
        predictions=model.binary_postprocess(model.predict(test_manager.get_df()))
    )
    test_manager.form_submit("submit.csv")
