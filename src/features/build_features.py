from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import json


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
    y = df["age"]
    X = df.drop("age", axis=1)
    return X, y


def main():
    data = load_interim_dataset()
    X, y = prepare_datasets(data, [])
    print(X.head())
    print(y.head())
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)


if __name__ == "__main__":
    main()
