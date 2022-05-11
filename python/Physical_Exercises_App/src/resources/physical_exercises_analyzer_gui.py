import os
import sys
import time
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.ttk import Style, Frame, Label, Radiobutton, Button, Checkbutton

import cv2

from resources.constants import PUSHUPS, PULLUPS, MANUAL, AUTO, TRUE, FALSE, OK, INCOMPLETE, WRONG
from resources.evaluation_options import EvaluationOptions
from resources.openpose_model_preparator import analyze_model

DISABLED = tk.DISABLED
NORMAL = tk.NORMAL

H1_FONT = "Helvetica 16 bold"
H2_FONT = "Helvetica 11"
H2_FONT_BOLD = "Helvetica 11 bold"

GREEN = "#03940d"
YELLOW = "#fcf63d"
RED = "#fc3d3d"
WHITE = "#ffffff"
BLACK = "#000000"
LIGHT_BLUE = "#c0f4fc"

WHITE_RGB = [255, 255, 255]
BLACK_RGB = [0, 0, 0]


def show_frame(title, frame):
    cv2.imshow(title, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        sys.exit(0)


def get_video_fps(file_path):
    video = cv2.VideoCapture(file_path)
    return video.get(cv2.CAP_PROP_FPS)


def process_frame(frame, evaluation_options, ex_eval, ok_count, incomplete_count, wrong_count, total_count):
    procent = 100
    width_orig = int(frame.shape[1] * procent / 100)
    width = int(frame.shape[1] * procent / 100 + 275)
    height = int(frame.shape[0] * procent / 100)
    dim = (width, height)
    dim_orig = (width_orig, height)
    frame_resized = cv2.resize(frame, dim)
    frame = cv2.resize(frame, dim_orig)
    distance = int(height / 7)

    cv2.copyMakeBorder(frame, 0, 0, 0, 275, cv2.BORDER_CONSTANT, frame_resized, WHITE_RGB)

    cv2.putText(frame_resized, evaluation_options.exercise_type, (width_orig + 15, distance), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, BLACK_RGB, 2)
    cv2.putText(frame_resized, ex_eval, (width_orig + 15, distance * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK_RGB, 2)
    cv2.putText(frame_resized, "OK: " + str(ok_count), (width_orig + 15, distance * 3), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                BLACK_RGB, 2)
    cv2.putText(frame_resized, "INCOMPLETE: " + str(incomplete_count), (width_orig + 15, distance * 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, BLACK_RGB, 2)
    cv2.putText(frame_resized, "WRONG: " + str(wrong_count), (width_orig + 15, distance * 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, BLACK_RGB, 2)
    cv2.putText(frame_resized, "TOTAL: " + str(total_count), (width_orig + 15, distance * 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, BLACK_RGB, 2)
    # show_frame(evaluation_options.filename, frame_resized)
    return frame_resized


class ExercisesAnalyzerApp:
    def __init__(self):
        self.__initialize_misc()

        self.__initialize_master()
        self.__initialize_styles()

        self.__initialize_left_frame()
        self.__initialize_right_frame()
        self.__pack_frames()

        self.master.mainloop()

    def __initialize_misc(self):
        self.evaluation_options = EvaluationOptions()
        self.json_path = ""
        self.filename_no_extension = ""
        self.python_folder_path = os.getcwd()[:-len("/Physical_Exercises_App/src")]
        self.openpose_python_path = self.python_folder_path + "/openpose_python.py"

    def __initialize_styles(self):
        self.radio_button_style = Style()
        self.radio_button_style.configure("TRadiobutton", font=H2_FONT, background=LIGHT_BLUE)

        self.check_button_style = Style()
        self.check_button_style.configure("TCheckbutton", font=H2_FONT, background=LIGHT_BLUE)

        self.button_style = Style()
        self.button_style.configure("TButton", font=H2_FONT_BOLD)

        self.frame_style = Style()
        self.frame_style.configure("TFrame", background=LIGHT_BLUE)

        self.label_style = Style()
        self.label_style.configure("TLabel", font=H1_FONT, background=LIGHT_BLUE)

    def __initialize_master(self):
        self.master = tk.Tk()
        self.master.title("Physical Exercises Analyzer")
        self.master.geometry("950x600")
        self.master.resizable(False, False)
        self.master.update()

    def __initialize_left_frame(self):
        self.left_frame = Frame(self.master)

        self.detection_button_variable = tk.StringVar(self.left_frame, AUTO)
        self.exercise_type_variable = tk.StringVar(self.left_frame, PUSHUPS)

        self.detection_type_label = Label(self.left_frame, text="Detection type")
        self.detection_type_label.grid(row=0, column=0, columnspan=3, padx=(25, 0), pady=(20, 10), sticky="w")

        self.auto_button = Radiobutton(self.left_frame, text="Automatic", variable=self.detection_button_variable,
                                       value=AUTO, command=self.__check_detection_button)
        self.auto_button.grid(row=1, column=0, padx=(25, 0), sticky="w")

        self.manual_button = Radiobutton(self.left_frame, text="Manual", variable=self.detection_button_variable,
                                         value=MANUAL, command=self.__check_detection_button)
        self.manual_button.grid(row=1, column=1, sticky="w")

        self.exercise_type_label = Label(self.left_frame, text="Exercise type")
        self.exercise_type_label.grid(row=2, column=0, columnspan=3, padx=(25, 0), pady=10, sticky="w")

        self.pushups_button = Radiobutton(self.left_frame, text="Pushups", variable=self.exercise_type_variable,
                                          value=PUSHUPS, state=DISABLED)
        self.pushups_button.grid(row=3, column=0, padx=(25, 0), sticky="w")

        self.pullups_button = Radiobutton(self.left_frame, text="Pullups", variable=self.exercise_type_variable,
                                          value=PULLUPS, state=DISABLED)
        self.pullups_button.grid(row=3, column=1, sticky="w")

        self.coming_soon_button = Radiobutton(self.left_frame, text="More coming soon", state=DISABLED)
        self.coming_soon_button.grid(row=3, column=2, sticky="w")

        self.analyzer_label = Label(self.left_frame, text="Analyzer result")
        self.analyzer_label.grid(row=4, column=0, columnspan=3, padx=(25, 0), pady=(45, 15), sticky="w")

        self.result_text_box = tk.Text(self.left_frame, borderwidth=2, font=H2_FONT, width=80, height=20)
        self.result_text_box.configure(state=DISABLED)
        self.result_text_box.grid(row=5, column=0, columnspan=3, padx=(15, 25), pady=5, sticky="w")

    def __initialize_right_frame(self):
        self.right_frame = Frame(self.master)

        self.open_file_label = Label(self.right_frame, text="Upload exercise")
        self.open_file_label.grid(row=0, column=0, columnspan=2, padx=10, pady=(25, 15))

        self.open_file_button = Button(self.right_frame, text="Load video", command=self.__load_video, padding=3)
        self.open_file_button.grid(row=1, column=0, padx=(5, 5), pady=10)

        self.open_json_button = Button(self.right_frame, text="Load JSON folder", command=self.__load_json_folder,
                                       padding=3)
        self.open_json_button.grid(row=1, column=1, padx=(5, 10), pady=10)

        self.show_graphs_checkbutton_variable = tk.StringVar(self.right_frame, FALSE)
        self.show_graphs_checkbutton = Checkbutton(self.right_frame, text="Show graphs",
                                                   variable=self.show_graphs_checkbutton_variable, onvalue=TRUE,
                                                   offvalue=FALSE)
        self.show_graphs_checkbutton.grid(row=2, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w")

        self.show_stats_checkbutton_variable = tk.StringVar(self.right_frame, TRUE)
        self.show_stats_checkbutton = Checkbutton(self.right_frame, text="Show stats for nerds",
                                                  variable=self.show_stats_checkbutton_variable, onvalue=TRUE,
                                                  offvalue=FALSE)
        self.show_stats_checkbutton.grid(row=3, column=0, columnspan=2, padx=10, pady=(5, 10), sticky="w")

        self.download_results_label = Label(self.right_frame, text="Download results")
        self.download_results_label.grid(row=4, column=0, columnspan=2, padx=10, pady=(40, 10))

        self.download_txt_result = Button(self.right_frame, text="Download .txt result", state=DISABLED,
                                          command=self.__save_txt_result, padding=3)
        self.download_txt_result.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        self.download_video_result = Button(self.right_frame, text="Download video result", state=DISABLED,
                                            command=self.__save_video_result, padding=3)
        self.download_video_result.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

        self.status_label = Label(self.right_frame, text="Status", font=H2_FONT_BOLD)
        self.status_label.grid(row=7, column=0, columnspan=2, padx=10, pady=(40, 10), sticky="w")

        self.status = Label(self.right_frame, text="Waiting for input", font=H2_FONT_BOLD, foreground=BLACK,
                            background=WHITE, padding=5, relief="solid")
        self.status.grid(row=8, column=0, columnspan=2, padx=10, sticky="w")

        self.credits_label = Label(self.right_frame, text="Physical Exercises Analyzer\nMade By Daniel Popescu",
                                   font=H2_FONT, background=LIGHT_BLUE)
        self.credits_label.grid(row=10, column=0, columnspan=2, padx=10, pady=(70, 10), sticky="se")

    def __modify_status(self, new_status=None, new_color=None):
        if new_status is None:
            new_status = self.status["text"]
        if new_color is None:
            new_color = self.status["background"]

        self.status.configure(text=new_status, background=new_color)
        self.status.update()

    def __check_detection_button(self):
        if self.detection_button_variable.get() == AUTO:
            self.__change_exercise_type_buttons_state(DISABLED)
        else:
            self.__change_exercise_type_buttons_state(NORMAL)

    def __change_exercise_type_buttons_state(self, state):
        self.pushups_button.configure(state=state)
        self.pullups_button.configure(state=state)

    def __pack_frames(self):
        self.left_frame.pack(side=tk.LEFT, fill="both")
        self.right_frame.pack(side=tk.RIGHT, fill="both")

    def __save_txt_result(self):
        file_types = [("Text File", "*.txt")]

        file = fd.asksaveasfilename(
            title="Save text file",
            initialdir="./",
            initialfile=self.filename_no_extension + "_text_result",
            defaultextension=".txt",
            filetypes=file_types
        )
        if file == "":
            return

        with open(file, "w") as f:
            f.write(self.result_text_box.get("1.0", "end-1c"))
        self.__modify_status("Text file saved", GREEN)

    def __save_video_result(self):
        file_types = [("AVI", "*.avi")]

        file = fd.asksaveasfilename(
            title="Save video file",
            initialdir="./",
            initialfile=self.filename_no_extension + "_video_result",
            defaultextension=".avi",
            filetypes=file_types
        )
        if file == "":
            return

        self.__modify_status("Processing video file", YELLOW)
        self.process_and_save_video(file)

        self.__modify_status("Video file saved", GREEN)

    def process_and_save_video(self, filename):
        results = self.evaluation_options.results
        ex_eval_list = results[0]
        times_list = results[1]
        total_count = 0
        ok_count = 0
        incomplete_count = 0
        wrong_count = 0
        video_stream = cv2.VideoCapture(self.evaluation_options.video_path)
        assert video_stream.isOpened()

        width = video_stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        output = cv2.VideoWriter(filename, fourcc, 20.0, (int(width) + 275, int(height)))

        frames_in_video = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_difference = frames_in_video - sum(times_list)
        times_list[-1] = times_list[-1] + frames_difference - 1

        for i in range(len(times_list)):
            for _ in range(times_list[i]):
                _, frame = video_stream.read()
                if frame is None:
                    break
                processed_frame = process_frame(frame, self.evaluation_options, ex_eval_list[i], ok_count,
                                                incomplete_count, wrong_count, total_count)
                output.write(processed_frame)

            if ex_eval_list[i] == OK:
                ok_count = ok_count + 1
            elif ex_eval_list[i] == INCOMPLETE:
                incomplete_count = incomplete_count + 1
            elif ex_eval_list[i] == WRONG:
                wrong_count = wrong_count + 1
            total_count = total_count + 1

        _, frame = video_stream.read()
        if frame is None:
            return
        processed_frame = process_frame(frame, self.evaluation_options, ex_eval_list[-1], ok_count, incomplete_count,
                                        wrong_count, total_count)
        output.write(processed_frame)

    def __load_json_folder(self):
        json_path = fd.askdirectory(
            title="Load JSON folder",
            initialdir="./"
        )

        if json_path == "":
            return

        start_time = time.clock()

        self.__clear_result_text()
        self.__update_result_text("Folder loaded: " + json_path)
        self.download_txt_result.configure(state=DISABLED)
        self.download_video_result.configure(state=DISABLED)

        self.__modify_status("Analyzing JSON", YELLOW)
        self.json_path = json_path
        self.__get_file_name(json_path)
        self.__populate_evaluate_options()

        analyze_model(self.evaluation_options)
        self.__update_result_text(self.evaluation_options.result_text)

        self.__modify_status("JSON analyzed", GREEN)
        self.download_txt_result.configure(state=NORMAL)

        if self.show_stats_checkbutton_variable.get() == TRUE:
            self.__update_result_text(f"Program execution time: {round(time.clock() - start_time, 2)} sec")

    def __get_file_name(self, folder_path):
        _, _, files = next(os.walk(folder_path))
        if len(files) > 0:
            filename = files[0].split(".")[0]
            end_seq = "_000000000000_keypoints"
            self.filename_no_extension = filename[:-len(end_seq)]

    def __load_video(self):
        file_types = (
            ("MP4 File", "*.mp4"),
            ("AVI File", "*.avi")
        )

        file_path = fd.askopenfilename(
            title="Load video",
            initialdir="./",
            filetypes=file_types)

        if file_path == "":
            return

        start_time = time.clock()

        self.__clear_result_text()
        self.__update_result_text("File loaded: " + file_path)
        self.download_txt_result.configure(state=DISABLED)
        self.download_video_result.configure(state=DISABLED)

        self.__modify_status("Analyzing video", YELLOW)
        self.__generate_json_files(file_path)
        self.__populate_evaluate_options(file_path)

        analyze_model(self.evaluation_options)
        self.__update_result_text(self.evaluation_options.result_text)

        self.__modify_status("Video analyzed", GREEN)
        self.download_txt_result.configure(state=NORMAL)
        self.download_video_result.configure(state=NORMAL)

        if self.show_stats_checkbutton_variable.get() == TRUE:
            self.__update_result_text(f"Program execution time: {round(time.clock() - start_time, 2)} sec")

    def __populate_evaluate_options(self, file_path=""):
        self.evaluation_options.filename = self.filename_no_extension
        self.evaluation_options.video_path = file_path
        self.evaluation_options.folder_path = self.json_path
        self.evaluation_options.python_folder_path = self.python_folder_path
        if file_path != "":
            self.evaluation_options.fps = get_video_fps(file_path)
        self.evaluation_options.show_graphs = self.show_graphs_checkbutton_variable.get()
        self.evaluation_options.detection_type = self.detection_button_variable.get()
        if self.evaluation_options.detection_type == AUTO:
            self.evaluation_options.exercise_type = AUTO
        else:
            self.evaluation_options.exercise_type = self.exercise_type_variable.get()
        self.evaluation_options.show_stats = self.show_stats_checkbutton_variable.get()
        self.evaluation_options.results = []

    def __generate_json_files(self, file_path):
        filename = os.path.basename(file_path)
        self.filename_no_extension = filename.split(".")[0]
        self.json_path = os.getcwd() + "/resources/JSON_FILES/" + self.filename_no_extension
        initial_directory = os.getcwd()
        os.chdir(self.python_folder_path)

        command = f"""python {self.openpose_python_path} --model_pose COCO --display 0 --render_pose 0 --video {file_path} --net_resolution -1x176 --face_net_resolution 320x320 --number_people_max 1 --write_json {self.json_path}"""
        if self.show_stats_checkbutton_variable.get() == TRUE:
            self.__update_result_text("Openpose command executed:\n\n" + command)

        os.system(command)
        os.chdir(initial_directory)

        if self.show_stats_checkbutton_variable.get() == TRUE:
            self.__update_result_text(f"Saved JSON files to {self.json_path}")

    def __update_result_text(self, text=""):
        self.result_text_box.configure(state=NORMAL)
        self.result_text_box.insert(tk.END, text + "\n\n")
        self.result_text_box.update()
        self.result_text_box.configure(state=DISABLED)

        print(text)

    def __clear_result_text(self):
        self.result_text_box.configure(state=NORMAL)
        self.result_text_box.delete("1.0", tk.END)
        self.result_text_box.update()
        self.result_text_box.configure(state=DISABLED)

        self.evaluation_options.result_text = ""
