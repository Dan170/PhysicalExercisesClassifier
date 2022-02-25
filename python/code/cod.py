import json

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from dtw import dtw
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


def main():
    path = r'C:\Users\Dani\Desktop\info\licenta\proiect\PhysicalExercisesClassifier\python'
    pushups_good_label = "pushups_good"
    pushups_bad_label = "pushups_bad"
    pullups_good_label = "pullups_good"
    pullups_bad_label = "pullups_bad"

    pushups_good = model_exercise(path + r'\goodpushups', 'good_pushups_example', pushups_good_label)
    pushups_bad = model_exercise(path + r'\badpushups', 'bad_pushups_example', pushups_bad_label)

    pullups_good = model_exercise(path + r'\goodpullups', 'good_pullups_example', pullups_good_label)
    pullups_bad = model_exercise(path + r'\badpullups', 'bad_pullups_example', pullups_bad_label)

    # plot_y_features(df_exercise_good)
    # plot_y_features(df_exercise_bad)
    # plot_comparison_y_features(pushups_good, pushups_bad, pushups_good_label, pushups_bad_label)
    # evaluated = evaluate_dtw_values(pushups_good, pushups_bad)

    plot_comparison_y_features(pushups_good, pushups_bad, pushups_good_label, pushups_bad_label)
    evaluated = evaluate_dtw_values(pushups_good, pushups_bad)

    median_value = evaluated.median()[0]
    print("median: ", median_value)
    if median_value < 10:
        print("exercise is very good")
    elif median_value < 15:
        print("exercise is good")
    elif median_value < 25:
        print("exercise is ok")
    elif median_value < 50:
        print("exercise is incomplete")
    else:
        print("wrong exercise")

    evaluated.plot()
    # plt.show()


def evaluate_dtw(df1, df2, feature):
    x1 = range(df1.shape[0])
    y1 = df1[feature].values
    x2 = range(df2.shape[0])
    y2 = df2[feature].values

    dtw_value = dtw(df1[feature], df2[feature])
    print("dtw_value for feature {} is {}".format(feature, dtw_value.normalizedDistance))

    return dtw_value.normalizedDistance


def evaluate_dtw_values(df1, df2):
    dtw_values = []
    for feature in df1.columns:
        if '_y' in feature:
            dtw_values.append(evaluate_dtw(df1, df2, feature))
    return pd.DataFrame(dtw_values)


def plot_comparison_y_features(df1, df2, label1, label2):
    fig = make_subplots(rows=3, cols=6, start_cell="top-left")
    r = 1
    c = 1
    X1 = pd.Series(range(df1.shape[0]))
    X2 = pd.Series(range(df2.shape[0]))
    for feature in df1.columns:
        if '_y' in feature:
            fig.add_trace(go.Scatter(x=X1, y=df1[feature], name=feature + "_" + label1), row=r, col=c)
            fig.add_trace(go.Scatter(x=X2, y=df2[feature], name=feature + "_" + label2), row=r, col=c)
            fig.update_xaxes(title_text=feature, row=r, col=c)
            if c < 6:
                c = c + 1
            else:
                c = 1
                r = r + 1
    fig.update_layout(title_text=label1 + " vs " + label2, width=2000, height=1000)
    fig.show()


def plot_y_features(df):
    fig = make_subplots(rows=3, cols=6, start_cell="top-left")
    r = 1
    c = 1
    X = pd.Series(range(df.shape[0]))
    for feature in df.columns:
        if '_y' in feature:
            fig.add_trace(go.Scatter(x=X, y=df[feature], name=feature),
                          row=r, col=c)
            fig.update_xaxes(title_text=feature, row=r, col=c)
            if c < 6:
                c = c + 1
            else:
                c = 1
                r = r + 1
    fig.update_layout(title_text="Exercise y-axis movements breakdown", width=2000, height=1000)
    fig.show()


def read_pose_values(path, file_name):
    # try:
    path, dirs, files = next(os.walk(path))
    df_output = pd.DataFrame()
    for i in range(len(files)):
        if i <= 9:
            fpath = path + '\\' + file_name + '_00000000000' + str(i) + '_keypoints.json'
            pose_sample = read_json(fpath)
        elif i <= 99:
            fpath = path + '\\' + file_name + '_0000000000' + str(i) + '_keypoints.json'
            pose_sample = read_json(fpath)
        else:
            fpath = path + '\\' + file_name + '_000000000' + str(i) + '_keypoints.json'
            pose_sample = read_json(fpath)

        df_output = df_output.append(pose_sample, ignore_index=True)
    return df_output


# except Exception as e:
#     print(e)

def read_json(path):
    with open(path) as json_data:
        data = json.load(json_data)
    return data["people"][0]


'''
Nose – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,
Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,
Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,
LAnkle – 13, Right Eye – 14, Left Eye – 15, Right Ear – 16,
Left Ear – 17, Background – 18
'''


def transform_and_transpose(pose_data, label):
    output = pd.DataFrame()
    for i in range(pose_data.shape[0] - 1):
        if len(pose_data["pose_keypoints_2d"]) > 0:
            output = output.append(pd.DataFrame(pose_data["pose_keypoints_2d"][i]).T)
    # drop confidence detection
    for y in range(2, output.shape[1], 3):
        output.drop(columns=[y], inplace=True)

    # rename columns
    output.columns = ['nose_x', 'nose_y', 'right_shoulder_x', 'right_shoulder_y', 'right_elbow_x', 'right_elbow_y',
                      'right_wrist_x', 'right_wrist_y', 'left_shoulder_x', 'left_shoulder_y', 'left_elbow_x',
                      'left_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_hip_x', 'right_hip_y', 'right_knee_x',
                      'right_knee_y',
                      'right_ankle_x', 'right_ankle_y', 'left_hip_x', 'left_hip_y', 'left_knee_x', 'left_knee_y',
                      'left_ankle_x', 'left_ankle_y', 'right_eye_x', 'right_eye_y', 'left_eye_x', 'left_eye_y',
                      'right_ear_x', 'right_ear_y', 'left_ear_x', 'left_ear_y', 'background_x', 'background_y']
    # interpolate 0 values
    # output.replace(0, np.nan, inplace=True)
    output.interpolate(method='linear', limit_direction='forward', inplace=True)

    return output


def model_exercise(json, file_name, label):
    df_raw = read_pose_values(json, file_name)
    df_modeled = transform_and_transpose(df_raw, label)
    df_modeled.to_pickle(label + ".pkl")
    # print(df_modeled)
    return df_modeled


if __name__ == '__main__':
    main()
