from sklearn.preprocessing import MinMaxScaler
from openpose_model_preparator import load_from_pickle, PUSHUPS_FRONT, PUSHUPS_LEFT, PUSHUPS_RIGHT


def main():
    pickle_front = load_from_pickle(PUSHUPS_FRONT + ".pkl")
    pickle_left = load_from_pickle(PUSHUPS_LEFT + ".pkl")
    pickle_left_2 = load_from_pickle(PUSHUPS_LEFT + "2.pkl")
    pickle_right = load_from_pickle(PUSHUPS_RIGHT + ".pkl")


if __name__ == '__main__':
    main()
