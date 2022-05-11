from resources.constants import TRUE, FALSE, MANUAL, PUSHUPS


class EvaluationOptions:
    def __init__(self, filename="", video_path="", folder_path="", python_folder_path="", fps=0, show_graphs=FALSE,
                 detection_type=MANUAL, exercise_type=PUSHUPS, result_text="", show_stats=TRUE, results=None):
        self.filename = filename
        self.video_path = video_path
        self.folder_path = folder_path
        self.python_folder_path = python_folder_path
        self.fps = fps
        self.show_graphs = show_graphs
        self.detection_type = detection_type
        self.exercise_type = exercise_type
        self.result_text = result_text
        self.show_stats = show_stats
        self.results = results
