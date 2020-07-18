import os

all_file = []


def get_all_file(raw_dir):
    all_file_list = os.listdir(raw_dir)
    for f in all_file_list:
        file_path = os.path.join(raw_dir, f)
        if os.path.isdir(file_path):
            get_all_file(file_path)
        all_file.append(file_path)

    return all_file


def get_all_video_list(raw_dir, action_list):
    video_list = {}
    files = get_all_file(raw_dir)
    for action in action_list:
        tmp_video_list = []
        for f in files:
            if action + '.mp4' in f and 'result' not in f:
                tmp_video_list.append(f)

        video_list.update({action: tmp_video_list})

    return video_list
