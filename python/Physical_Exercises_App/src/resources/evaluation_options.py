from resources.constants import TRUE, MANUAL, PUSHUPS


class EvaluationOptions:
    def __init__(self, filename="", folder_path="", python_folder_path="", fps=0, show_graphs=TRUE,
                 detection_type=MANUAL, exercise_type=PUSHUPS):
        self.filename = filename
        self.folder_path = folder_path
        self.python_folder_path = python_folder_path
        self.fps = fps
        self.show_graphs = show_graphs
        self.detection_type = detection_type
        self.exercise_type = exercise_type
