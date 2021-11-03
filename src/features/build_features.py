import pandas as pd
import numpy as np
import json


def load_interim_dataset(file_path=r"D:\PROJEKTY\intonat.dl\data\interim\data.json"):
    with open(file_path, 'r') as fp:
        dataset_json = json.load(fp)
        dataset_json.pop("fundamental_frequency")
        return pd.DataFrame.from_dict(dataset_json)


def prepare_datasets(data, features):
    df = data.drop(["name", "sr"], axis=1)
    df = df.set_index(['age']).apply(pd.Series.explode).reset_index()
    statistics = [np.mean, np.std]
    for feature in features:
        for statistic in statistics:
            df[f"{feature}_{str(str(np.std).split(' ')[1])}"] = df[feature].map(lambda x: statistic(x))
    y = df["age"]
    X = df.drop("age", axis=1)
    print(df.head())
    return X, y


def main():
    data = load_interim_dataset()
    X, y = prepare_datasets(data, ["mfcc, delta_mfcc, delta2_mfcc, hnr"])


if __name__ == "__main__":
    main()
