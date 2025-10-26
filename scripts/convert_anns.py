from pathlib import Path
import datetime
import re
import os
import cv2
import numpy as np
import json

def time_str_to_seconds(time_str: str):
    hours, minutes, seconds, milliseconds = [
        int(segment)
        for segment
        in re.split(r"\.|\:", time_str)
    ]
    if hours > 0:
        print("Number of fours greater than zero.")

    return minutes * 60 + seconds + float(f"0.{milliseconds}")

if __name__ == "__main__":
    annotation_dir = Path("dataset") / "annotations"
    video_dir = Path("dataset") / "videos" / "mp4"
    output_dir = Path("dataset") / "annotations_processed"

    ann_filenames = [Path(fn).stem for fn in os.listdir(annotation_dir) if fn.startswith("2025-")]

    filename2labels = dict()
    for fn in ann_filenames:
        video_path = (video_dir) / f"{fn}.mp4"
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {fn}, fps {fps}, total frames: {total_frames}")


        with open(annotation_dir / f"{fn}.txt", "r") as fp:
            lines = fp.read().splitlines()

        frame_labels = np.zeros(total_frames).astype(np.int64)
        for idx, l in enumerate(lines):
            if len(l) == 0:
                print("skipped")
                continue
            [start, end] = l.split(" ")
            start_frame_idx, end_frame_idx = int(time_str_to_seconds(start) * fps), int(time_str_to_seconds(end) * fps)
            frame_labels[start_frame_idx:end_frame_idx] = idx + 1

            print(f"Processed line {idx}, start: {start}, end: {end}, start frame: {start_frame_idx}, end frame: {end_frame_idx}")
        
        filename2labels[fn] = frame_labels
    
    np.savez(output_dir, **filename2labels)
