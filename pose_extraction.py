import cv2
from pytube import YouTube
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# import torch
from tqdm import tqdm
from evaluate.coco_eval import get_multiplier, get_outputs

import argparse

parser = argparse.ArgumentParser(description='')


parser.add_argument('--source_video', default='./data/source/', help='Source video name')
parser.add_argument('--source_path', default='./data/source/', help='Path where the source video is saved')
parser.add_argument('--frames', default=1000, type=int, help='Path where the source video is saved')

args = parser.parse_args()

def extract_img_from_video(source_path, source_video):
    """
    Extracts the frames of vieo and saves it in 
    a directory

    Args:
        save_dir (Path): Directory where frames will be saved

    Returns:
        *bool: The return value. True for success, False otherwise.

    """

    save_dir = Path(source_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_dir = save_dir.joinpath('images')
    img_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(save_dir.joinpath(source_video)))
    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = args.frames if args.frames < max_frames else max_frames

    print('Extracting images from video')
    for i in tqdm(range(frames)):
        _, frame = cap.read()
        
        # Cropped the image into a square and resize it to (512,512)
        oh = (frame.shape[0] - shape_dst) // 2
        ow = (frame.shape[1] - shape_dst) // 2
        frame = img[oh:oh+shape_dst, ow:ow+shape_dst]
        frame = cv2.resize(frame, (512, 512))
        cv2.imwrite(str(img_dir.joinpath(f'img_{i:04d}.png')), frame)

def main(args):
    extract_img_from_video(args.source_path, args.source_video)

if __name__ == "__main__":
    main(args)
    # pass
