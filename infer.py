import os
import argparse
import numpy as np
import cv2
import torch
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader

import glob

from tqdm import tqdm
from network.models import model_selection
from data_list import ImageList
from video_to_images import get_cropped_face

torch.autograd.set_detect_anomaly(True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# TODO make note!
# Assumption: Video file name is of the form "{int}.mp4"!!!!!!!!!!!!!!!!!!!!!!!!


def create_images(video_dir, image_dir, interval):
        paths = np.array(glob.glob(video_dir + "/infer/*.mp4"))
        os.makedirs(image_dir, exist_ok=True)
        
        image_subdir = os.path.join(image_dir, "infer")
        os.makedirs(image_subdir, exist_ok=True)

        for path in tqdm(paths):
            f = open(os.path.join(image_dir, "infer_dir" + '.txt'), 'a+')
            video_id = path.split('/')[-1].split('.')[0]
            os.makedirs(os.path.join(image_subdir, video_id), exist_ok=True)
            reader = cv2.VideoCapture(path)
            count = 0
            frame_num = 0
            max_frames = reader.get(cv2.CAP_PROP_FRAME_COUNT)
            img_folder = os.path.join(image_subdir, video_id)
            f.write("%s %s\n" % (img_folder, video_id[-5:]))
            f.close()

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
                        
                        image_path = os.path.join(img_folder, image_id)
                        # save cropped image
                        cv2.imwrite(image_path, cropped_face)
                        count += 1


def get_dataloader(image_dir):

    img_size = 299
    mean = [0.5] * 3
    std = [0.5] * 3
    normalize_transform = Normalize(mean, std)

    infer = open(os.path.join(image_dir, "infer_dir.txt")).readlines()
    inferset = ImageList(infer, normalize_transform, img_size)

    dataloader = DataLoader(inferset, batch_size=1)

    return dataloader


def evaluate(net, dataloader, threshold, max_images):
    net.eval()
    f = open('results.txt', 'w')
    with torch.no_grad():
        for batch_idx, (inputs, video_id) in enumerate(dataloader):
            inputs = inputs.squeeze(0)
            # trim to preven GPU out of memeory
            if inputs.shape[0] > max_images:
                inputs = inputs[:max_images]
            inputs = inputs.to(device)
 
            outputs = net(inputs)
            prediction_imgs = outputs.argmax(1)
            
            prediction = 'fake' if prediction_imgs.float().mean() >= threshold else 'real'

            f.write("Video ID ending in {}: {}".format(video_id.item(), prediction))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_dir', '-i', type=str, default='videos')
    parser.add_argument('--image_dir', '-o', type=str, default='images')
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--max_images', type=int, default=200)
    parser.add_argument('--threshold', type=float, default=0.5)

    args = parser.parse_args()
    
    create_images(args.video_dir, args.image_dir, args.interval)

    dataloader = get_dataloader(args.image_dir)

    net, *_ = model_selection(modelname='xception', num_out_classes=2)
    net = net.to(device)
    net.load_state_dict(torch.load('weights/xception.pth'))

    evaluate(net, dataloader, args.threshold, args.max_images)

