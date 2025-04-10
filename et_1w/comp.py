import json

with open("et_164k_txt_1w.json", "r") as f:
    all_file = json.load(f)

def calculate_time_segments_ratio(duration, tgt):
    if len(tgt) % 2 != 0:
        raise ValueError("tgt list")

    total_segments_time = 0.0
    for i in range(0, len(tgt), 2):
        start = tgt[i]
        end = tgt[i + 1]
        segment_duration = end - start
        total_segments_time += segment_duration

    ratio = int(duration / total_segments_time)
    print(ratio)
    return ratio

ratio = [0,] * 100

for i in all_file:
    ratio[calculate_time_segments_ratio(i['duration'], i['tgt'])] += 1

print(ratio)