"""
Extract images from videos in Celeb-DF v2

Author: HanChen
Date: 12.10.2020
"""
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from extract_crop import extract_video, extract_video_halve

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main():
    global index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    model = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)

    video_root_path = '/data1/cby/dataset/UADFV'
    image_root_path = '/raid/chenhan/UADFV'
    video_sub_path = 'video/real'
    if not os.path.isdir(image_root_path):
        os.mkdir(image_root_path)
    for name in video_sub_path.split('/'):
        image_root_path = os.path.join(image_root_path, name)
        if not os.path.isdir(image_root_path):
            os.mkdir(image_root_path)
        video_root_path = os.path.join(video_root_path, name)

    wrong_txt = open('wrong4.txt', 'w')

    wrong_videos = []
    for index, video in tqdm(enumerate(os.listdir(video_root_path))):
        input_path = os.path.join(video_root_path, video)
        output_path = os.path.join(image_root_path, video)
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        try:
            extract_video_halve(input_path, output_path, model)
        except:
            print('halve : ' + video)
            wrong_videos.append(video)
            continue

    for video in wrong_videos:
        input_path = os.path.join(video_root_path, video)
        output_path = os.path.join(image_root_path, video)
        try:
            extract_video(input_path, output_path, model)
        except:
            print('whole : ' + video)
            wrong_txt.write(video + '\n')
            continue
    print(video_sub_path + ' have ' + str(index + 1) + ' videos')


if __name__ == "__main__":
    main()
