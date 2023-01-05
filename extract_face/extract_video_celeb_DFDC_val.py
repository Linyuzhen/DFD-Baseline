"""
Extract images from videos in Celeb-DF v2

Author: HanChen
Date: 2.11.2020
"""
import torch
import pandas as pd
from tqdm import tqdm
from facenet_pytorch import MTCNN
from extract_crop import extract_video, extract_video_halve

import shutil
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main():
    global index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    model = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)

    video_root_path = '/data1/cby/dataset/DFDC/videos/Val'
    image_root_path = '/raid/chenhan/DFDC-face_Val'
    if not os.path.isdir(image_root_path):
        os.mkdir(image_root_path)
    csv_name = 'labels.csv'
    csv_file = pd.read_csv(os.path.join(video_root_path, csv_name), names=None)
    shutil.copy(os.path.join(video_root_path, csv_name), os.path.join(image_root_path, csv_name))

    wrong_txt = open('wrong.txt', 'w')
    wrong_videos = []

    for index, video in tqdm(enumerate(csv_file['filename'])):
        video_name = os.path.join('part_all', video)
        if not os.path.isdir(os.path.join(image_root_path, 'part_all')):
            os.mkdir(os.path.join(image_root_path, 'part_all'))

        input_path = os.path.join(video_root_path, video_name)
        output_path = os.path.join(image_root_path, video_name)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        try:
            extract_video_halve(input_path, output_path, model)
        except:
            print('halve : ' + video_name)
            wrong_videos.append(video_name)

    for video_name in wrong_videos:
        input_path = os.path.join(video_root_path, video_name)
        output_path = os.path.join(image_root_path, video_name)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        try:
            extract_video(input_path, output_path, model)
        except:
            print('whole : ' + video_name)
            wrong_txt.write(video_name + '\n')
    print('part_all ' + ' have ' + str(index + 1) + ' videos')


if __name__ == "__main__":
    main()
