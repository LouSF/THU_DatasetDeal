import os
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

dataset_jsonfile_org = "/home/fangbenhao/L/DatasetS/ET-Instruct-164K/et_instruct_164k_vid.json"
dataset_video_path = "/home/fangbenhao/L/DatasetS/ET-Instruct-164K/videos"
dataset_jsonfile_save = "/home/fangbenhao/L/DatasetS/ET-Instruct-164K"

def get_video_duration_ffprobe(filepath):
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            filepath
        ]
        output = subprocess.check_output(cmd).decode('utf-8')
        data = json.loads(output)
        return float(data['format']['duration'])
    except Exception as e:
        print(f"\nError processing {filepath}: {str(e)}")
        return None


def process_item(data):

    if data["task"] not in ["tal", "tvg"]:
        return None

    if data["task"] == "tvg" and data.get("source") not in ["didemo", "tacos"]:
        return None

    video_file = os.path.join(dataset_video_path, data["video"])
    if not os.path.exists(video_file):
        return None

    duration = get_video_duration_ffprobe(video_file)
    if duration is None or not (60 <= duration <= 180):
        return None

    return data


if __name__ == "__main__":
    with open(dataset_jsonfile_org, "r") as f:
        json_data = json.load(f)

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        with tqdm(
                total=len(json_data),
                desc="Processing Videos",
                unit="video",
                dynamic_ncols=True
        ) as pbar:
            futures = []
            for data in json_data:
                future = executor.submit(process_item, data)
                future.add_done_callback(lambda _: pbar.update(1))
                futures.append(future)

            results = []
            for future in futures:
                result = future.result()
                if result is not None:
                    results.append(result)

    json_tal = [x for x in results if x["task"] == "tal"]
    json_tvg = [x for x in results if x["task"] == "tvg"]

    json_all = json_tal + json_tvg
    random.shuffle(json_all)

    with open(os.path.join(dataset_jsonfile_save, "et_instruct_164k_vid_spilt.json"), "w") as f:
        json.dump(json_all, f, indent=4)

    print(f"\nFinal counts:")
    print(f"tvg_manual: {len(json_tvg)}")
    print(f"tal: {len(json_tal)}")
    print(f"Total valid videos: {len(results)}/{len(json_data)}")

#Final counts:
# tvg_manual: 13699
# tal: 13110
# Total valid videos: 26809/163880
