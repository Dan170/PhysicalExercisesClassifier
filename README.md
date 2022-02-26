# PhysicalExercisesClassifier

How to run:

`python .\openpose_python.py --model_pose COCO --net_resolution "320x176" --face_net_resolution "320x320" --part_candidates --write_json`

`python .\openpose_python.py --model_pose COCO --video "./examples/media/bad_pushups_example.mp4" --net_resolution "320x176" --face_net_resolution "320x320" --keypoint_scale 3 --number_people_max 1 --write_json "/bad"`

`python .\openpose_python.py --model_pose COCO --video "./examples/media/bad_pullups_example.mp4" --net_resolution "320x176" --face_net_resolution "320x320" --number_people_max 1 --write_json "/bad_pullups"`