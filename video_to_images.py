
import os
import argparse
import numpy as np
import cv2
import dlib

import glob
import random
from tqdm import tqdm

from sklearn.model_selection import train_test_split

face_detector = dlib.get_frontal_face_detector()

def get_boundingbox(face, width, height, scale=1.3, minsize=None):

    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def get_paths(video_dir, seed):

        all_paths = np.array(glob.glob(video_dir + "/*/*.mp4"))
        random.shuffle(all_paths)

        train_paths, test_paths = train_test_split(all_paths, test_size=0.15, random_state=seed)
        train_paths, val_paths = train_test_split(train_paths, test_size=0.15, random_state=seed)
        
        return {"train":train_paths, "val":val_paths, "test":test_paths}

def create_images(video_dir, image_dir, interval, seed):

    path_dict = get_paths(video_dir, seed)
    os.makedirs(image_dir, exist_ok=True)
    
    for split, paths in path_dict.items():

        image_subdir = os.path.join(image_dir, split)
        os.makedirs(image_subdir, exist_ok=True)

        for path in tqdm(paths):
            f = open(os.path.join(image_dir, split + '.txt'), 'a+')
            label = 1 if 'fake' in path else 0
            video_id = path.split('/')[-1].split('.')[0]
            os.makedirs(os.path.join(image_subdir, video_id), exist_ok=True)
            reader = cv2.VideoCapture(path)
            count = 0
            frame_num = 0
            max_frames = reader.get(cv2.CAP_PROP_FRAME_COUNT)
            print("video: ", video_id)
            print("Total frames", max_frames)
            while reader.isOpened():
                success, image = reader.read()
                frame_num += 1
                if frame_num >= max_frames:
                    break
                
                if frame_num % interval != 0:
                    continue
                
                if success:
                    cropped_face = get_cropped_face(image)
                    if cropped_face is not None:
                        # print(count)
                        image_id = str(count) + '.png'
                        image_path = os.path.join(image_subdir, video_id, image_id)
                        f.write("%s %s\n" % (image_path, str(label)))
                        # save cropped image
                        cv2.imwrite(image_path, cropped_face)
                        count += 1
            
            f.close()
    

def get_cropped_face(image):

    height, width = image.shape[:2]

    # Detect with dlib  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)

    if len(faces):
        face = faces[0]
        # Face crop with dlib and bounding box scale enlargement
        x, y, size = get_boundingbox(face, width, height)
        cropped_face = image[y:y+size, x:x+size].copy()

        return cropped_face


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_dir', '-i', type=str, default='videos')
    parser.add_argument('--image_dir', '-o', type=str, default='images')
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=2021)

    args = parser.parse_args()
    random.seed(args.seed)

    create_images(args.video_dir, args.image_dir, args.interval, args.seed)