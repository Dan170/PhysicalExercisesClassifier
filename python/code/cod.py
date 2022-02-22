import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def main():
    df_exercise_1 = model_exercise(r'C:\Users\Dani\Desktop\info\licenta\proiect\PhysicalExercisesClassifier\python\1', '<label>')
    print(df_exercise_1)


def read_pose_values(path):
    try:
        os.path.isdir(path)
        path, dirs, files = next(os.walk(path))
        df_output = pd.DataFrame()
        for i in range(len(files)):
            if i <= 9:
                pose_sample = pd.read_json(
                    path_or_buf=path + '\\' + '00000000000' + str(i) + '_keypoints.json', typ='series')
            elif i <= 99:
                pose_sample = pd.read_json(
                    path_or_buf=path + '\\' + '0000000000' + str(i) + '_keypoints.json', typ='series')
            else:
                pose_sample = pd.read_json(
                    path_or_buf=path + '\\' + '000000000' + str(i) + '_keypoints.json', typ='series')
            df_output = df_output.append(pose_sample, ignore_index=True)
        return df_output
    except Exception as e:
        print(e)


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
        if len(pose_data.people[i]) > 0:
            output = output.append(pd.DataFrame(pose_data.people[i][0]['pose_keypoints']).T)
    # drop confidence detection
    for y in range(2, output.shape[1], 3):
        output.drop(columns=[y], inplace=True)
        # rename columns
        output.columns = ['nose_x', 'nose_y', 'right_shoulder_x', 'right_shoulder_y', 'right_elbow_x', 'right_elbow_y',
                          'right_wrist_x', 'right_wrist_y', 'left_shoulder_x', 'left_shoulder_y', 'left_elbow_x',
                          'left_elbow_y',
                          'left_wrist_x', 'left_wrist_y', 'right_hip_x', 'right_hip_y', 'right_knee_x', 'right_knee_y',
                          'right_ankle_x', 'right_ankle_y', 'left_hip_x', 'left_hip_y', 'left_knee_x', 'left_knee_y',
                          'left_ankle_x', 'left_ankle_y', 'right_eye_x', 'right_eye_y', 'left_eye_x', 'left_eye_y',
                          'right_ear_x', 'right_ear_y', 'left_ear_x', 'left_ear_y', 'background_x', 'background_y']

        # interpolate 0 values
        output.replace(0, np.nan, inplace=True)
        output.interpolate(method='linear', limit_direction='forward', inplace=True)

    return output


def model_exercise(json, label):
    df_raw = read_pose_values(json)
    return transform_and_transpose(df_raw, label)


if __name__ == '__main__':
    main()
