from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from openpose_model_preparator import load_from_pickle, split_dataframe, PUSHUPS_FRONT, PUSHUPS_LEFT, PUSHUPS_RIGHT, \
    MULTIPLE
import numpy as np


def main():
    pickle_models = __read_pickle_models()
    split_pickle_models = __split_pickle_models(pickle_models)
    normalized_split_models = __normalize_split_models(split_pickle_models)

    print


def __normalize_model(model):
    scaler = MinMaxScaler()
    imputer = SimpleImputer()

    numpy_model = scaler.fit_transform(model)
    return imputer.fit_transform(numpy_model)


def __normalize_split_models(split_pickle_models):
    normalized_split_models = []

    for split_model in split_pickle_models:
        split_model = __normalize_model(split_model)
        normalized_split_models.append(split_model)

    return normalized_split_models


def __split_pickle_models(pickle_models):
    split_pickle_models = []
    for model in pickle_models:
        split_dataframes = split_dataframe(model)
        for split_df in split_dataframes:
            split_pickle_models.append(split_df)
    return split_pickle_models


def __read_pickle_models():
    pickle_front_multiple = load_from_pickle(PUSHUPS_FRONT + MULTIPLE + ".pkl")
    pickle_left_multiple = load_from_pickle(PUSHUPS_LEFT + MULTIPLE + ".pkl")
    pickle_left_2_multiple = load_from_pickle(PUSHUPS_LEFT + "2" + MULTIPLE + ".pkl")
    pickle_right = load_from_pickle(PUSHUPS_RIGHT + ".pkl")

    pickle_models = [pickle_front_multiple, pickle_left_multiple, pickle_left_2_multiple, pickle_right]
    return pickle_models


if __name__ == '__main__':
    main()
