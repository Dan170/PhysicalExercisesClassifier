# PhysicalExercisesClassifier

How to run:

`python {openpose_python.py_path} --model_pose COCO --video {video_path} --net_resolution "-1x176" --face_net_resolution "320x320" --number_people_max 1 --write_json {json_folder_path}`

With no display output:

`python {openpose_python.py_path} --model_pose COCO --display 0 --render_pose 0 --video {video_path} --net_resolution "-1x176" --face_net_resolution "320x320" --number_people_max 1 --write_json {json_folder_path} --write_video {output_video_path}`

Check [OpenPose Flags Doc](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/include/openpose/flags.hpp) for more info on run commands