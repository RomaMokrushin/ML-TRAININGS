import lightgbm as lgb

def binary_postprocess(predictions):
    return [int(i >= 0.5) for i in predictions]

def df_to_lbm(df, label):
    return lgb.Dataset(data=df, label=label)