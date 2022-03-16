import tkinter as tk
from tkinter.ttk import Style, Frame, Label, Radiobutton, Button

from resources.openpose_model_preparator import PUSHUPS, PULLUPS

AUTO = "AUTO"
MANUAL = "MANUAL"
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


class ExercisesAnalyzerApp:
    def __init__(self):
        self.__initialize_master()
        self.__initialize_styles()

        self.__initialize_left_frame()
        self.__initialize_right_frame()

        self.__pack_frames()
        self.master.mainloop()

    def __initialize_styles(self):
        self.radio_button_style = Style()
        self.radio_button_style.configure("TRadiobutton", font=H2_FONT, background=LIGHT_BLUE)

        self.button_style = Style()
        self.button_style.configure("TButton", font=H2_FONT_BOLD)

        self.frame_style = Style()
        self.frame_style.configure("TFrame", background=LIGHT_BLUE)

        self.label_style = Style()
        self.label_style.configure("TLabel", font=H1_FONT, background=LIGHT_BLUE)

    def __initialize_master(self):
        self.master = tk.Tk()
        self.master.title("Physical Exercises Analyzer")
        self.master.geometry("900x600")
        self.master.resizable(False, False)
        self.master.update()

    def __initialize_left_frame(self):
        self.left_frame = Frame(self.master)

        self.detection_button_variable = tk.StringVar(self.left_frame, MANUAL)
        self.exercise_type_variable = tk.StringVar(self.left_frame, PUSHUPS)

        self.detection_type_label = Label(self.left_frame, text="Detection type:")
        self.detection_type_label.grid(row=0, column=0, columnspan=3, padx=(25, 0), pady=(20, 10), sticky="w")

        self.auto_button = Radiobutton(self.left_frame, text="Automatic", variable=self.detection_button_variable,
                                       value=AUTO, command=self.__check_detection_button)
        self.auto_button.grid(row=1, column=0, padx=(25, 0), sticky="w")

        self.manual_button = Radiobutton(self.left_frame, text="Manual", variable=self.detection_button_variable,
                                         value=MANUAL, command=self.__check_detection_button)
        self.manual_button.grid(row=1, column=1, sticky="w")

        self.exercise_type_label = Label(self.left_frame, text="Exercise type:")
        self.exercise_type_label.grid(row=2, column=0, columnspan=3, padx=(25, 0), pady=10, sticky="w")

        self.pushups_button = Radiobutton(self.left_frame, text="Pushups", variable=self.exercise_type_variable,
                                          value=PUSHUPS)
        self.pushups_button.grid(row=3, column=0, padx=(25, 0), sticky="w")

        self.pullups_button = Radiobutton(self.left_frame, text="Pullups", variable=self.exercise_type_variable,
                                          value=PULLUPS)
        self.pullups_button.grid(row=3, column=1, sticky="w")

        self.coming_soon_button = Radiobutton(self.left_frame, text="More coming soon", state=DISABLED)
        self.coming_soon_button.grid(row=3, column=2, sticky="w")

        self.analyzer_label = Label(self.left_frame, text="Analyzer result:")
        self.analyzer_label.grid(row=4, column=0, columnspan=3, padx=(25, 0), pady=(45, 15), sticky="w")

        self.result_text_box = tk.Text(self.left_frame, borderwidth=2, font=H2_FONT, width=70, height=20)
        self.result_text_box.configure(state=DISABLED)
        self.result_text_box.grid(row=5, column=0, columnspan=3, padx=(15, 55), pady=5, sticky="w")

    def __initialize_right_frame(self):
        self.right_frame = Frame(self.master)

        self.open_file_label = Label(self.right_frame, text="Upload exercise video:")
        self.open_file_label.grid(row=0, column=0, padx=15, pady=(25, 15))

        self.open_file_button = Button(self.right_frame, text="Load Video")
        self.open_file_button.grid(row=1, column=0, padx=15, pady=10)

        self.download_results_label = Label(self.right_frame, text="Download results:")
        self.download_results_label.grid(row=2, column=0, padx=15, pady=(40, 10))

        self.download_txt_result = Button(self.right_frame, text="Download .txt result")
        self.download_txt_result.grid(row=3, column=0, padx=15, pady=10)

        self.download_video_result = Button(self.right_frame, text="Download video result")
        self.download_video_result.grid(row=4, column=0, padx=15, pady=10)

        self.status_label = Label(self.right_frame, text="Status:", font=H2_FONT_BOLD)
        self.status_label.grid(row=5, column=0, padx=15, pady=(40, 10), sticky="w")

        self.status = Label(self.right_frame, text="Waiting for input", font=H2_FONT_BOLD, foreground=GREEN,
                            background=WHITE, padding=5, relief="solid")
        self.status.grid(row=6, column=0, padx=15, sticky="w")

        self.credits_label = Label(self.right_frame, text="Physical Exercises Analyzer\nMade By Daniel Popescu",
                                   font=H2_FONT, background=LIGHT_BLUE)
        self.credits_label.grid(row=10, column=0, columnspan=2, padx=15, pady=(155, 10), sticky="se")

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
