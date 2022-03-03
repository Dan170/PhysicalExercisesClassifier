import json

import pandas as pd
import os
import numpy as np
from dtw import dtw
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

MAX_ROWS = 5
MAX_COLUMNS = 4
MAX_WIDTH = 1500
MAX_HEIGHT = 1200

KEYPOINTS_JSON = '_keypoints.json'

PUSHUPS_COLUMNS_TO_DROP = ['wrist', 'elbow', 'ankle']
PULLUPS_COLUMNS_TO_DROP = ['wrist', 'elbow']
EMPTY_COLUMNS_TO_DROP = []

'''
Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4, Left Shoulder – 5, Left Elbow – 6, 
Left Wrist – 7, Right Hip – 8, Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12, Left Ankle – 13, 
Right Eye – 14, Left Eye – 15, Right Ear – 16, Left Ear – 17, Background – 18
'''
POSE_MODEL_COLUMNS = ['nose_x', 'nose_y', 'right_shoulder_x', 'right_shoulder_y', 'right_elbow_x', 'right_elbow_y',
                      'right_wrist_x', 'right_wrist_y', 'left_shoulder_x', 'left_shoulder_y', 'left_elbow_x',
                      'left_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_hip_x', 'right_hip_y', 'right_knee_x',
                      'right_knee_y', 'right_ankle_x', 'right_ankle_y', 'left_hip_x', 'left_hip_y', 'left_knee_x',
                      'left_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_eye_x', 'right_eye_y', 'left_eye_x',
                      'left_eye_y', 'right_ear_x', 'right_ear_y', 'left_ear_x', 'left_ear_y', 'background_x',
                      'background_y']


def main():
    path = r'C:\Users\Dani\Desktop\info\licenta\proiect\PhysicalExercisesClassifier\python'
    pushups_good_label = "pushups_good"
    pushups_bad_label = "pushups_bad"
    pullups_good_label = "pullups_good"
    pullups_bad_label = "pullups_bad"

    pushups_exercise = "pushups"
    pullups_exercise = "pullups"

    pushups_good = model_prepare(path + r'\goodpushups', 'good_pushups_example', pushups_good_label, False,
                                 pushups_exercise)
    pushups_bad = model_prepare(path + r'\badpushups', 'bad_pushups_example', pushups_bad_label, False,
                                pushups_exercise)

    pullups_good = model_prepare(path + r'\goodpullups', 'good_pullups_example', pullups_good_label, False,
                                 pullups_exercise)
    pullups_bad = model_prepare(path + r'\badpullups', 'bad_pullups_example', pullups_bad_label, False,
                                pullups_exercise)

    # plot_y_features(df_exercise_good)
    # plot_y_features(df_exercise_bad)
    # plot_comparison_y_features(pushups_good, pushups_bad, pushups_good_label, pushups_bad_label)
    # evaluated = evaluate_dtw_values(pushups_good, pushups_bad)

    plot_compare_dataframes(pushups_good, pushups_bad, pushups_good_label, pushups_bad_label)
    evaluated = evaluate_dtw_columns(pushups_good, pushups_bad)

    median_value = round(evaluated.median()[0])
    print("median: ", median_value)
    if median_value < 10:
        print("exercise is very good")
    elif median_value < 15:
        print("exercise is good")
    elif median_value < 25:
        print("exercise is ok")
    elif median_value < 35:
        print("exercise is incomplete")
    else:
        print("wrong exercise")


def evaluate_dynamic_time_warping(df1, df2, feature):
    dtw_value = dtw(df1[feature], df2[feature])
    print("dtw normalized distance for feature {} is {}".format(feature, dtw_value.normalizedDistance))

    return dtw_value.normalizedDistance


def evaluate_dtw_columns(df1, df2):
    dtw_values = []
    for column in df1.columns:
        if '_y' in column and column in df2.columns:
            dtw_values.append(evaluate_dynamic_time_warping(df1, df2, column))
    return pd.DataFrame(dtw_values)


def plot_compare_dataframes(df1, df2, label1, label2):
    current_row = 1
    current_column = 1

    figure = make_subplots(rows=MAX_ROWS, cols=MAX_COLUMNS, start_cell="top-left")
    df1_series = pd.Series(range(df1.shape[0]))
    df2_series = pd.Series(range(df2.shape[0]))
    for column in df1.columns:
        if '_y' in column:
            figure.add_trace(go.Scatter(x=df1_series, y=df1[column], name=column + "_" + label1), row=current_row,
                             col=current_column)
            figure.add_trace(go.Scatter(x=df2_series, y=df2[column], name=column + "_" + label2), row=current_row,
                             col=current_column)
            figure.update_xaxes(title_text=column, row=current_row, col=current_column)
            if current_column < MAX_COLUMNS:
                current_column = current_column + 1
            else:
                current_column = 1
                current_row = current_row + 1
    figure.update_layout(title_text=label1 + " vs " + label2, width=MAX_WIDTH, height=MAX_HEIGHT)
    figure.show()


def plot_dataframe(df, label):
    current_row = 1
    current_column = 1

    figure = make_subplots(rows=MAX_ROWS, cols=MAX_COLUMNS, start_cell="top-left")
    df_series = pd.Series(range(df.shape[0]))
    for column in df.columns:
        if '_y' in column:
            figure.add_trace(go.Scatter(x=df_series, y=df[column], name=column + column), row=current_row,
                             col=current_column)
            figure.update_xaxes(title_text=column, row=current_row, col=current_column)
            if current_column < MAX_COLUMNS:
                current_column = current_column + 1
            else:
                current_column = 1
                current_row = current_row + 1
    figure.update_layout(title_text=label, width=MAX_WIDTH, height=MAX_HEIGHT)
    figure.show()


def create_pose_dataframe(folder_path, file_name):
    try:
        _, _, files = next(os.walk(folder_path))
        dataframe = pd.DataFrame()
        for i in range(len(files)):
            if i <= 9:
                file_path = folder_path + '\\' + file_name + '_00000000000' + str(i) + KEYPOINTS_JSON
                json_data = read_json_file(file_path)
            elif i <= 99:
                file_path = folder_path + '\\' + file_name + '_0000000000' + str(i) + KEYPOINTS_JSON
                json_data = read_json_file(file_path)
            elif i <= 999:
                file_path = folder_path + '\\' + file_name + '_000000000' + str(i) + KEYPOINTS_JSON
                json_data = read_json_file(file_path)
            else:
                file_path = folder_path + '\\' + file_name + '_00000000' + str(i) + KEYPOINTS_JSON
                json_data = read_json_file(file_path)

            dataframe = dataframe.append(json_data, ignore_index=True)
        return dataframe
    except Exception as ex:
        print(ex)


def read_json_file(file_path):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
    return json_data["people"][0]


def transform_and_transpose(raw_pd_pose_model, exercise_type):
    modeled_pd_pose_model = pd.DataFrame()
    model_rows_count = raw_pd_pose_model.shape[0]
    for i in range(model_rows_count - 1):
        if len(raw_pd_pose_model["pose_keypoints_2d"]) > 0:
            modeled_pd_pose_model = modeled_pd_pose_model.append(
                pd.DataFrame(raw_pd_pose_model["pose_keypoints_2d"][i]).T)

    drop_confidence_columns(modeled_pd_pose_model)

    modeled_pd_pose_model.columns = POSE_MODEL_COLUMNS
    switch_exercise_type_drop_columns(modeled_pd_pose_model, exercise_type)
    drop_zero_predominant_columns(modeled_pd_pose_model)

    modeled_pd_pose_model.replace(0, np.nan, inplace=True)
    modeled_pd_pose_model.interpolate(method='linear', limit_direction='forward', inplace=True)

    return modeled_pd_pose_model


def switch_exercise_type_drop_columns(modeled_pd_pose_model, exercise_type):
    if exercise_type == 'pushups':
        drop_specific_columns(modeled_pd_pose_model, PUSHUPS_COLUMNS_TO_DROP)
    elif exercise_type == 'pullups':
        drop_specific_columns(modeled_pd_pose_model, PULLUPS_COLUMNS_TO_DROP)
    else:
        drop_specific_columns(modeled_pd_pose_model, EMPTY_COLUMNS_TO_DROP)


def drop_specific_columns(modeled_pd_pose_model, columns_to_drop):
    for column_to_drop in columns_to_drop:
        for column in modeled_pd_pose_model.columns:
            if column_to_drop in column:
                modeled_pd_pose_model.drop(columns=[column], inplace=True)


def drop_zero_predominant_columns(modeled_pd_pose_model):
    rows_count = modeled_pd_pose_model.shape[0]
    columns_to_drop = []
    for column in modeled_pd_pose_model.columns:
        if (modeled_pd_pose_model[column] == 0).sum() >= rows_count / 2:
            column_to_drop = column[:-2]
            columns_to_drop.append(column_to_drop)

    columns_to_drop = list(set(columns_to_drop))
    for column in columns_to_drop:
        modeled_pd_pose_model.drop(columns=[column + "_x"], inplace=True)
        modeled_pd_pose_model.drop(columns=[column + "_y"], inplace=True)


def drop_confidence_columns(modeled_pd_pose_model):
    model_columns_count = modeled_pd_pose_model.shape[1]
    for i in range(2, model_columns_count, 3):
        modeled_pd_pose_model.drop(columns=[i], inplace=True)


def model_prepare(folder_path, file_name, label, save_model, exercise_type):
    raw_pd_pose_model = create_pose_dataframe(folder_path, file_name)
    modeled_pd_pose_model = transform_and_transpose(raw_pd_pose_model, exercise_type)
    if save_model:
        save_to_pickle(modeled_pd_pose_model, label)
    return modeled_pd_pose_model


def save_to_pickle(modeled_pd_pose_model, label):
    modeled_pd_pose_model.to_pickle(label + ".pkl")
    print("model ", label, " saved to ", label + ".pkl")


def load_from_pickle(label):
    return pd.read_pickle(label + ".pkl")


if __name__ == '__main__':
    main()
