from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler


def load_interim_dataset(file_path=r"D:\PROJEKTY\intonat.dl\data\interim\data.json"):
    with open(file_path, 'r') as fp:
        dataset_json = json.load(fp)
        if len(dataset_json["fundamental_frequency"]):
            dataset_json.pop("fundamental_frequency")
        return pd.DataFrame.from_dict(dataset_json)


def prepare_datasets(data, features):
    df = data.drop(["sr", "name"], axis=1)
    # getting rid of rows with different elements counts
    df = df.drop(df[df["mfcc"].map(lambda x: len(x)) != df["hnr"].map(lambda x: len(x))].index)
    df = df.explode(df.drop(["age", ], axis=1).columns.values.tolist()).reset_index(drop=True)
    statistics = [np.mean, np.std]
    for feature in features:
        for statistic in statistics:
            df[f"{feature}_{str(str(statistic).split(' ')[1])}"] = df[feature].map(lambda x: statistic(x))
    df.to_feather(r"D:\PROJEKTY\intonat.dl\data\interim\dataset.ftr")
    bins = pd.IntervalIndex.from_tuples([(x, x+10) for x in range(19,97,10)], closed='left')
    df["age"] = pd.cut(df["age"], bins)
    y = df["age"]
    X = df.drop("age", axis=1)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y


def load_data():
    data = load_interim_dataset()
    X, y = prepare_datasets(data, ["mfcc", "delta_mfcc", "delta2_mfcc"])
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)
    return X_train, y_train, X_test, y_test


def main():
    X_train, y_train, X_test, y_test = load_data()
    print(X_train.head())


if __name__ == "__main__":
    main()
