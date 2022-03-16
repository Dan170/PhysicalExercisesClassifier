import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Style

from openpose_model_preparator import PUSHUPS, PULLUPS

AUTO = "AUTO"
MANUAL = "MANUAL"

H1_FONT = "Helvetica 16 bold"
H2_FONT = "Helvetica 11"


class ExercisesAnalyzerApp:
    def __init__(self):
        self.__initialize_master()

        self.radio_button_style = Style(self.master)
        self.radio_button_style.configure("TRadiobutton", font=H2_FONT)

        self.__initialize_left_frame()
        self.__initialize_frames()

        self.__initialize_text_box()

        self.__pack_frames()
        self.master.mainloop()

    def __initialize_master(self):
        self.master = tk.Tk()
        self.master.title("Physical Exercises Analyzer App")
        self.master.geometry("1200x600")
        self.master.resizable(False, False)
        self.master.update()
        self.master_width = self.master.winfo_width()
        self.master_height = self.master.winfo_height()

    def __initialize_left_frame(self):
        self.left_frame = ttk.Frame(self.master, height=self.master_height, width=self.master_width * 2 / 3)

        self.detection_button_variable = tk.StringVar(self.left_frame, MANUAL)
        self.exercise_type_variable = tk.StringVar(self.left_frame, PUSHUPS)

        self.detection_type_label = ttk.Label(self.left_frame, text="Detection type:", font=H1_FONT)
        self.detection_type_label.grid(row=0, column=0, padx=25, pady=10, sticky="w")

        self.auto_button = ttk.Radiobutton(self.left_frame, text="Automatic", variable=self.detection_button_variable,
                                           value=AUTO, command=self.__check_detection_button)
        self.auto_button.grid(row=1, column=0, padx=25, sticky="w")
        self.manual_button = ttk.Radiobutton(self.left_frame, text="Manual", variable=self.detection_button_variable,
                                             value=MANUAL, command=self.__check_detection_button)
        self.manual_button.grid(row=1, column=1, padx=5, sticky="w")

        self.exercise_type_label = ttk.Label(self.left_frame, text="Exercise type:", font=H1_FONT)
        self.exercise_type_label.grid(row=2, column=0, padx=25, pady=10, sticky="w")

        self.pushups_button = ttk.Radiobutton(self.left_frame, text="Pushups", variable=self.exercise_type_variable,
                                              value=PUSHUPS)
        self.pushups_button.grid(row=3, column=0, padx=25, sticky="w")
        self.pullups_button = ttk.Radiobutton(self.left_frame, text="Pullups", variable=self.exercise_type_variable,
                                              value=PULLUPS)
        self.pullups_button.grid(row=3, column=1, padx=5, sticky="w")

    def __initialize_frames(self):
        self.right_frame = ttk.Frame(self.master, height=self.master_height, width=self.master_width * 1 / 3)
        self.text_box_frame = ttk.Frame(self.left_frame, height=self.master_height / 2,
                                        width=self.left_frame.winfo_width())

    def __initialize_text_box(self):
        self.text_box = tk.Text(self.text_box_frame)
        self.text_box.insert(tk.END, "Exercises Analyzer Result:\n")
        self.text_box.configure(state="disabled")
        self.text_box.grid(padx=10, pady=10)

    def __check_detection_button(self):
        if self.detection_button_variable.get() == AUTO:
            self.__change_exercise_type_buttons_state("disabled")
        else:
            self.__change_exercise_type_buttons_state("normal")

    def __change_exercise_type_buttons_state(self, state):
        self.pushups_button.configure(state=state)
        self.pullups_button.configure(state=state)

    def __pack_frames(self):
        self.left_frame.pack(side=tk.LEFT, fill="both")
        self.right_frame.pack(side=tk.RIGHT, fill="both")
