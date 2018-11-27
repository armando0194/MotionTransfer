import cv2
from pytube import YouTube
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from OpenPoseModel import OpenPoseModel
from Pix2PixModel import Pix2PixModel
from post import *
import argparse
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# tf.logging.set_verbosity(tf.logging.ERROR)

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
    
    return max_frames

def generate_labels(path, open_pose_weights, frames, save_dir):
    """
    Given frames it detects the pose of the human inside

    Args:
        save_dir (Path): Directory where frames will be saved

    Returns:
        *bool: The return value. True for success, False otherwise.

    """
    path = Path(path)
    
    img_dir = path.joinpath('images')
    img_dir.mkdir(exist_ok=True)
    save_dir = path.joinpath(save_dir)
    save_dir.mkdir(exist_ok=True)

    model = OpenPoseModel('test', open_pose_weights)

    for idx in tqdm(range(3038, frames)):
        img_path = img_dir.joinpath(f'img_{idx:04d}.png')
        img = cv2.imread(str(img_path))
        label = model.predict(img)
        
        label = cv2.cvtColor(label, cv2.COLOR_GRAY2BGR)
        img = np.concatenate((img, label), axis=1)

        cv2.imwrite(str(save_dir.joinpath(f'{idx:04d}.png')), img)
    
def main(args):
    # print(args.source_path)
    # print(args.source_video)
    # extract_frames_from_video(args.source_path, args.source_video, args.frames)
    # generate_labels(args.source_path, args.open_pose_weights, args.frames, 'labels')
    
    # max_frames = extract_frames_from_video(args.target_path, args.target_video, args.frames)
    # max_frames = 6693
    # generate_labels(args.target_path, args.open_pose_weights, max_frames, 'train_images')
    model = Pix2PixModel(args)
    model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--source_video', default='source.mp4', help='Source video name')
    parser.add_argument('--source_path', default='../data/source/', help='Path where the source video is saved')
    parser.add_argument('--target_video', default='target.mp4', help='target video name')
    parser.add_argument('--target_path', default='../data/target/', help='Path where the target video is saved')
    parser.add_argument('--frames', default=100000, type=int, help='Number of frames to process')
    parser.add_argument('--open_pose_weights', type=str, default='../model/openpose/model.h5', help='path to the weights file')
    parser.add_argument('--seed', type=str, default=None, help='Seed')
    parser.add_argument('--p2p_output_dir', type=str, default='../model/pix2pix/', help='path to the weights file')
    parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
    parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
    parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
    parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
    parser.set_defaults(flip=True)
    parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")
    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
    parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")
    parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
    parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
    parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
    parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
    parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")
    parser.add_argument("--which_direction", type=str, default="BtoA", choices=["AtoB", "BtoA"])


    args = parser.parse_args()
    main(args)

