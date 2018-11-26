import cv2
from pytube import YouTube
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from OpenPoseModel import OpenPoseModel
from post import *
import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument('--source_video', default='source.mp4', help='Source video name')
parser.add_argument('--source_path', default='../data/source/', help='Path where the source video is saved')
parser.add_argument('--target_video', default='target.mp4', help='target video name')
parser.add_argument('--target_path', default='../data/target/', help='Path where the target video is saved')
parser.add_argument('--frames', default=1000, type=int, help='Number of frames to process')
parser.add_argument('--open_pose_weights', type=str, default='../model/keras/model.h5', help='path to the weights file')

args = parser.parse_args()

def extract_frames_from_video(source_path, source_video, frames):
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
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = frames if frames < video_frames else video_frames

    print('Extracting images from video')
    for i in tqdm(range(max_frames)):
        _, frame = cap.read()
        shape_dst = np.min(frame.shape[:2])
        # Cropped the image into a square and resize it to (512,512)
        oh = (frame.shape[0] - shape_dst) // 2
        ow = (frame.shape[1] - shape_dst) // 2
        frame = frame[oh:oh+shape_dst, ow:ow+shape_dst]
        frame = cv2.resize(frame, (512, 512))
        cv2.imwrite(str(img_dir.joinpath(f'img_{i:04d}.png')), frame)

def generate_labels(path, open_pose_weights, frames, save_dir):
    path = Path(path)
    
    img_dir = path.joinpath('images')
    img_dir.mkdir(exist_ok=True)
    save_dir = path.joinpath(save_dir)
    save_dir.mkdir(exist_ok=True)

    model = OpenPoseModel('test', open_pose_weights)

    for idx in tqdm(range(117, 117+4)):
        img_path = img_dir.joinpath(f'img_{idx:04d}.png')
        img = cv2.imread(str(img_path))
        label = model.predict(img)
        
        # plt.imshow(label)
        # plt.show()
        # cv2.imwrite(str(test_img_dir.joinpath(f'img_{idx:04d}.png')), img)
        
        cv2.imwrite(str(save_dir.joinpath(f'label_{idx:04d}.png')), label[:,:,-1])
    
def main(args):
    # print(args.source_path)
    # print(args.source_video)
    # extract_frames_from_video(args.source_path, args.source_video, args.frames)
    generate_labels(args.source_path, args.open_pose_weights, args.frames, 'labels')
    
    # extract_frames_from_video(args.target_path, args.target_video, args.frames)
    # generate_labels(args.target_path, args.open_pose_weights, args.frames, 'labels')

if __name__ == "__main__":
    main(args)
