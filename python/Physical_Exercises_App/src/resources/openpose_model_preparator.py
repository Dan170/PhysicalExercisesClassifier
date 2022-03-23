import json

import pandas as pd
import os
import numpy as np
from dtw import dtw
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from resources.constants import PUSHUPS, ASCENDING, DESCENDING, LEFT, RIGHT, PULLUPS, PUSHUPS_LEFT, PUSHUPS_RIGHT, \
    AUTO, TRUE
from resources.evaluation_options import EvaluationOptions

KEYPOINTS_JSON = '_keypoints.json'
PATH = "JSON_FILES\\"

MAX_ROWS = 5
MAX_COLUMNS = 4
MAX_WIDTH = 1500
MAX_HEIGHT = 1200

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


def analyze_model(evaluation_options):
    dataframe, side_detected, exercise_type = model_prepare(evaluation_options)

    if exercise_type is PULLUPS:
        side_detected = PULLUPS
    pickle_models_path = evaluation_options.python_folder_path + "/Physical_Exercises_App/src/resources/pickle_models/"
    correct_dataframe, correct_file_name = __load_correct_model(pickle_models_path, side_detected, evaluation_options)

    if evaluation_options.show_graphs == TRUE:
        plot_compare_dataframes(correct_dataframe, dataframe, correct_file_name + "_CORRECT",
                                evaluation_options.filename)

    dataframes = split_dataframe(dataframe)
    __update_result_text(evaluation_options, f"Found {len(dataframes)} {exercise_type}")

    for index, split_df in enumerate(dataframes):
        __update_result_text(evaluation_options, f"||||||||||||  For {exercise_type} number {index + 1}")

        if evaluation_options.show_graphs == TRUE and len(dataframes) > 1:
            plot_compare_dataframes(correct_dataframe, split_df, correct_file_name + "_CORRECT",
                                    evaluation_options.filename + str(index + 1))
        evaluate_dataframe(split_df, correct_dataframe, evaluation_options)

        if evaluation_options.fps != 0:
            exec_time = split_df.shape[0] / evaluation_options.fps
            __update_result_text(evaluation_options, "Execution time: {:.2f} seconds".format(exec_time))


def evaluate_dataframe(dataframe, correct_dataframe, evaluation_options=EvaluationOptions()):
    evaluated_dataframe = __evaluate_dtw_columns(dataframe, correct_dataframe)

    median_value = round(evaluated_dataframe.median()[0])
    __update_result_text(evaluation_options, f"Repetition threshold: {median_value}")
    if median_value < 25:
        __update_result_text(evaluation_options, "Exercise is ok")
    elif median_value < 35:
        __update_result_text(evaluation_options, "Exercise is incomplete")
    else:
        __update_result_text(evaluation_options, "Wrong exercise")


def model_prepare(options, save_model=False):
    folder_path = options.folder_path
    file_name = options.filename
    detection_type = options.detection_type

    raw_pd_pose_model = __create_pose_dataframe(folder_path, file_name)
    modeled_pd_pose_model, side_detected = __transform_model(raw_pd_pose_model)
    __interpolate_model(modeled_pd_pose_model)

    dataframe_for_ml = split_dataframe(modeled_pd_pose_model, options)[0]

    if detection_type is AUTO:
        exercise_type = find_exercise_type(dataframe_for_ml)
    else:
        exercise_type = options.exercise_type

    __find_exercise_type_drop_columns(modeled_pd_pose_model, exercise_type)

    if save_model:
        __save_to_pickle(modeled_pd_pose_model, file_name)
    return modeled_pd_pose_model, side_detected, exercise_type


def find_exercise_type(dataframe_for_ml):
    exercise_type = PUSHUPS
    return exercise_type


def split_dataframe(dataframe, options=EvaluationOptions()):
    dataframe_values = dataframe["nose_y"].values
    dataframe_length = len(dataframe_values)
    previous_value = dataframe_values[0]
    starting_index = 0
    graph_type = __get_graph_type(dataframe_values)
    __update_result_text(options, f"Found graph type: {graph_type}")
    found_extreme = False
    dataframes = []

    for index, current_value in enumerate(dataframe_values):
        if graph_type is ASCENDING:
            # first if clause is checking if the values are still increasing. After that, we suppose that the values
            # started decreasing
            if found_extreme is False and (
                    current_value >= previous_value or (dataframe_length > index + 1 and (
                    current_value <= dataframe_values[index + 1]
                    or __check_abs(current_value, dataframe_values[index + 1])))):
                previous_value = current_value
                continue

            if dataframe_length > index + 1 and \
                    (current_value > dataframe_values[index + 1]
                     or __check_abs(current_value, dataframe_values[index + 1])):
                found_extreme = True
                previous_value = current_value
            elif index + 1 is dataframe_length:
                split_df = dataframe.iloc[starting_index:index + 1, :]
                dataframes.append(split_df)
                break
            else:
                split_df = dataframe.iloc[starting_index:index + 1, :]
                dataframes.append(split_df)
                starting_index = index
                found_extreme = False
                previous_value = current_value

    return dataframes


def plot_compare_dataframes(df1, df2, label1, label2):
    current_row = 1
    current_column = 1

    figure = make_subplots(rows=MAX_ROWS, cols=MAX_COLUMNS, start_cell="top-left")
    df1_series = pd.Series(range(df1.shape[0]))
    df2_series = pd.Series(range(df2.shape[0]))
    for column in df1.columns:
        if '_y' in column and column in df2.columns:
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

    title = label1 + " vs " + label2
    figure.update_layout(title_text=title, width=MAX_WIDTH, height=MAX_HEIGHT, hovermode="x")

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


def load_from_pickle(file_path="", evaluation_options=EvaluationOptions()):
    __update_result_text(evaluation_options, f"Loaded {file_path}")
    return pd.read_pickle(file_path)


def __update_result_text(evaluation_options, text):
    evaluation_options.result_text = evaluation_options.result_text + text + "\n\n"
    print(text)


def __check_abs(value1, value2):
    return abs(round(value1) - round(value2)) <= 2


def __get_graph_type(dataframe_values):
    starting_value = dataframe_values[0]
    dataframe_length = len(dataframe_values)

    if dataframe_length > 15:
        if starting_value > dataframe_values[15]:
            return DESCENDING
        else:
            return ASCENDING


def __evaluate_dynamic_time_warping(df1, df2, feature):
    dtw_value = dtw(df1[feature], df2[feature])
    # print("DTW normalized distance for {} is {}".format(feature, dtw_value.normalizedDistance))

    return dtw_value.normalizedDistance


def __evaluate_dtw_columns(df1, df2):
    dtw_values = []
    for column in df1.columns:
        if '_y' in column and column in df2.columns:
            dtw_values.append(__evaluate_dynamic_time_warping(df1, df2, column))
    return pd.DataFrame(dtw_values)


def __read_json_file(file_path):
    with open(file_path) as json_file:
        json_data = json.load(json_file)
    if "people" in json_data and len(json_data["people"]) > 0:
        return json_data["people"][0]


def __create_pose_dataframe(folder_path, file_name):
    try:
        _, _, files = next(os.walk(folder_path))
        dataframe = pd.DataFrame()
        for i in range(len(files)):
            if i <= 9:
                file_path = folder_path + '\\' + file_name + '_00000000000' + str(i) + KEYPOINTS_JSON
                json_data = __read_json_file(file_path)
            elif i <= 99:
                file_path = folder_path + '\\' + file_name + '_0000000000' + str(i) + KEYPOINTS_JSON
                json_data = __read_json_file(file_path)
            elif i <= 999:
                file_path = folder_path + '\\' + file_name + '_000000000' + str(i) + KEYPOINTS_JSON
                json_data = __read_json_file(file_path)
            else:
                file_path = folder_path + '\\' + file_name + '_00000000' + str(i) + KEYPOINTS_JSON
                json_data = __read_json_file(file_path)

            dataframe = dataframe.append(json_data, ignore_index=True)
        return dataframe
    except Exception as ex:
        print(ex)


def __interpolate_model(modeled_pd_pose_model):
    modeled_pd_pose_model.replace(0, np.nan, inplace=True)
    modeled_pd_pose_model.interpolate(method='linear', limit_direction='forward', inplace=True)


def __find_side(modeled_pd_pose_model):
    left_ear_count = (modeled_pd_pose_model["left_ear_y"] != 0).sum()
    right_ear_count = (modeled_pd_pose_model["right_ear_y"] != 0).sum()
    if left_ear_count > right_ear_count:
        return LEFT
    else:
        return RIGHT


def __drop_specific_columns(modeled_pd_pose_model, columns_to_drop):
    for column_to_drop in columns_to_drop:
        for column in modeled_pd_pose_model.columns:
            if column_to_drop in column:
                modeled_pd_pose_model.drop(columns=[column], inplace=True)


def __drop_zero_predominant_columns(modeled_pd_pose_model):
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


def __drop_confidence_columns(modeled_pd_pose_model):
    model_columns_count = modeled_pd_pose_model.shape[1]
    for i in range(2, model_columns_count, 3):
        modeled_pd_pose_model.drop(columns=[i], inplace=True)


def __save_to_pickle(modeled_pd_pose_model, file_name):
    modeled_pd_pose_model.to_pickle("./pickle_models/" + file_name + ".pkl")
    print("model ", file_name, " saved to ", file_name + ".pkl")


def __find_exercise_type_drop_columns(modeled_pd_pose_model, exercise_type):
    if exercise_type == PUSHUPS:
        __drop_specific_columns(modeled_pd_pose_model, PUSHUPS_COLUMNS_TO_DROP)
    elif exercise_type == PULLUPS:
        __drop_specific_columns(modeled_pd_pose_model, PULLUPS_COLUMNS_TO_DROP)
    else:
        __drop_specific_columns(modeled_pd_pose_model, EMPTY_COLUMNS_TO_DROP)


def __transform_model(raw_pd_pose_model):
    modeled_pd_pose_model = pd.DataFrame()
    model_rows_count = raw_pd_pose_model.shape[0]
    for i in range(model_rows_count - 1):
        if len(raw_pd_pose_model["pose_keypoints_2d"]) > 0:
            modeled_pd_pose_model = modeled_pd_pose_model.append(
                pd.DataFrame(raw_pd_pose_model["pose_keypoints_2d"][i]).T)

    __drop_confidence_columns(modeled_pd_pose_model)
    modeled_pd_pose_model.columns = POSE_MODEL_COLUMNS
    side_detected = __find_side(modeled_pd_pose_model)
    __drop_zero_predominant_columns(modeled_pd_pose_model)

    return modeled_pd_pose_model, side_detected


def __load_correct_model(path, side_detected, evaluation_options):
    __update_result_text(evaluation_options, f"Side detected: {side_detected}")
    if side_detected is PULLUPS:
        return load_from_pickle(path + PULLUPS + ".pkl", evaluation_options), PULLUPS
    elif side_detected is LEFT:
        return load_from_pickle(path + PUSHUPS_LEFT + ".pkl", evaluation_options), PUSHUPS_LEFT
    elif side_detected is RIGHT:
        return load_from_pickle(path + PUSHUPS_RIGHT + ".pkl", evaluation_options), PUSHUPS_RIGHT
