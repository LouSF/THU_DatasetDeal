import os
import sys
import json
import subprocess
import multiprocessing
from tqdm import tqdm
from typing import Optional

dataset_base_path = '/yeesuanAI10/thumt/loushengfeng/Datasets/star'
video_base_path_list = os.path.join(dataset_base_path, 'videos')

json_file_path_list = [
    'test.json',
    'train.json',
    'val.json',
]

dataset_table_star = {
    "id": "question_id",
    "video_path": "video_id",
    "video_length": None,
}

dataset_table_list = {
    "star": dataset_table_star,
}


# def main(json_path):
#     with open(json_path) as f:
#         videos = json.load(f)
#
#     if not all({'path', 'duration'} <= set(v) for v in videos):
#         raise ValueError("Invalid JSON format. Missing required fields (path/duration)")
#
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         results = []
#         with tqdm(total=len(videos), desc="Processing Videos") as pbar:
#             for result in pool.imap(process_video, videos):
#                 results.append(result)
#                 pbar.update()
#
#     success_count = 0
#     failed_count = 0
#     max_diff = 0
#     discrepancies = []
#
#     for res in results:
#         if res['status'] == 'success':
#             success_count += 1
#             diff = abs(res['diff'])
#             if diff > 0.1:
#                 discrepancies.append(res)
#             if diff > max_diff:
#                 max_diff = diff
#         else:
#             failed_count += 1


def get_duration(path) -> Optional[bool]:
    if not os.path.exists(path):
        return False
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        print(output)
        return True
    except Exception as e:
        print(f"\nError processing {path}: {str(e)}")
        return False


def process_video(video) -> Optional[bool]:

    video_path = video[dataset_table_star['video_path']]+'.mp4'
    video_path = os.path.abspath(video_path)

    if get_duration(video_path):
        return True
    else :
        return False


def main():
    json_rec = []
    for file_name in json_file_path_list:
        json_file_path = os.path.join(dataset_base_path, file_name)
        try:
            with open(json_file_path, "r") as f:
                json_rec.extend(json.load(f))
                print("json loading finished")
        except Exception as e:
            print(f"\nError processing {json_file_path}: {str(e)}")
            print("json file error")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = []
        passed = 0
        unpass = 0
        with tqdm(total=len(json_rec), desc="Processing Videos") as pbar:
            for result in pool.imap(process_video, json_rec):
                results.append(result)
                if result:
                    passed += 1
                else:
                    unpass += 1

                pbar.update()

        print(f"passed {passed}")
        print(f"unpass {unpass}")



if __name__ == "__main__":
    main()
