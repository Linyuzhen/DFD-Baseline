import cv2
import numpy as np
import json
import random
from tqdm import tqdm
from copy import deepcopy
from skimage.metrics import structural_similarity
from skimage.io import imread
from skimage import transform

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def main(image_root_path, real_sub_path, deepfake_sub_path, real_test_videos, fake_test_videos):

    for idx in tqdm(range(len(real_test_videos))):
        real_video_path = os.path.join(image_root_path, real_sub_path, real_test_videos[idx])
        temp_save_root_path = '/'
        for name in real_video_path.replace('youtube', 'youtube_ssim_diff').split('/'):
            temp_save_root_path = os.path.join(temp_save_root_path, name)
            if not os.path.isdir(temp_save_root_path):
                os.mkdir(temp_save_root_path)
        temp_save_root_path = '/'
        for name in real_video_path.replace('youtube', 'youtube_abs_diff').split('/'):
            temp_save_root_path = os.path.join(temp_save_root_path, name)
            if not os.path.isdir(temp_save_root_path):
                os.mkdir(temp_save_root_path)
        image_names = os.listdir(real_video_path)
        random.shuffle(image_names)

        deepfake_video_path = os.path.join(image_root_path, deepfake_sub_path, fake_test_videos[idx])
        if os.path.isdir(deepfake_video_path):
            if len(os.listdir(deepfake_video_path)):
                temp_save_root_path = '/'
                for name in deepfake_video_path.replace('Deepfakes', 'Deepfakes_ssim_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)
                temp_save_root_path = '/'
                for name in deepfake_video_path.replace('Deepfakes', 'Deepfakes_abs_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)
						
                image_names = os.listdir(deepfake_video_path)
                for image_name in image_names:
                    try:
                        img_pairs = []
                        image_path1 = os.path.join(deepfake_video_path, image_name)
                        image_path2 = os.path.join(real_video_path, image_name)
                        image1, image2 = cv2.imread(image_path1, cv2.IMREAD_COLOR), cv2.imread(image_path2, cv2.IMREAD_COLOR)
                        image1 = image1.astype(np.float32)
                        image2 = image2.astype(np.float32)
                        temp = np.abs(image1 - image2)
                        cv2.imwrite(os.path.join(temp_save_root_path, image_name), temp.astype(np.uint8))
                        d, a = structural_similarity(image1, image2, multichannel=True, data_range=255, full=True)
                        a = 1 - a
                        temp = (a * 255.0).astype(np.uint8)
                        cv2.imwrite(os.path.join(temp_save_root_path.replace('Deepfakes_abs_diff', 'Deepfakes_ssim_diff'), image_name), temp)
                    except:
                        continue

        Face2Face_video_path = os.path.join(image_root_path, deepfake_sub_path.replace('Deepfakes', 'Face2Face'),
                                            fake_test_videos[idx])
        if os.path.isdir(Face2Face_video_path):
            if len(os.listdir(Face2Face_video_path)):
                temp_save_root_path = '/'
                for name in Face2Face_video_path.replace('Face2Face', 'Face2Face_ssim_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)
                temp_save_root_path = '/'
                for name in Face2Face_video_path.replace('Face2Face', 'Face2Face_abs_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)

                image_names = os.listdir(Face2Face_video_path)
                for image_name in image_names:
                    try:
                        img_pairs = []
                        image_path1 = os.path.join(Face2Face_video_path, image_name)
                        image_path2 = os.path.join(real_video_path, image_name)
                        image1, image2 = cv2.imread(image_path1, cv2.IMREAD_COLOR), cv2.imread(image_path2, cv2.IMREAD_COLOR)
                        image1 = image1.astype(np.float32)
                        image2 = image2.astype(np.float32)
                        temp = np.abs(image1 - image2)
                        cv2.imwrite(os.path.join(temp_save_root_path, image_name), temp.astype(np.uint8))
                        d, a = structural_similarity(image1, image2, multichannel=True, data_range=255, full=True)
                        a = 1 - a
                        temp = (a * 255.0).astype(np.uint8)
                        cv2.imwrite(os.path.join(temp_save_root_path.replace('Face2Face_abs_diff', 'Face2Face_ssim_diff'), image_name), temp)
                    except:
                        continue

        FaceSwap_video_path = os.path.join(image_root_path, deepfake_sub_path.replace('Deepfakes', 'FaceSwap'),
                                           fake_test_videos[idx])
        if os.path.isdir(FaceSwap_video_path):
            if len(os.listdir(FaceSwap_video_path)):
                temp_save_root_path = '/'
                for name in FaceSwap_video_path.replace('FaceSwap', 'FaceSwap_ssim_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)
                temp_save_root_path = '/'
                for name in FaceSwap_video_path.replace('FaceSwap', 'FaceSwap_abs_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)

                image_names = os.listdir(FaceSwap_video_path)
                for image_name in image_names:
                    try:
                        img_pairs = []
                        image_path1 = os.path.join(FaceSwap_video_path, image_name)
                        image_path2 = os.path.join(real_video_path, image_name)
                        image1, image2 = cv2.imread(image_path1, cv2.IMREAD_COLOR), cv2.imread(image_path2, cv2.IMREAD_COLOR)
                        image1 = image1.astype(np.float32)
                        image2 = image2.astype(np.float32)
                        temp = np.abs(image1 - image2)
                        cv2.imwrite(os.path.join(temp_save_root_path, image_name), temp.astype(np.uint8))
                        d, a = structural_similarity(image1, image2, multichannel=True, data_range=255, full=True)
                        a = 1 - a
                        temp = (a * 255.0).astype(np.uint8)
                        cv2.imwrite(os.path.join(temp_save_root_path.replace('FaceSwap_abs_diff', 'FaceSwap_ssim_diff'), image_name), temp)
                    except:
                        continue


        NeuralTextures_video_path = os.path.join(image_root_path,
                                                 deepfake_sub_path.replace('Deepfakes', 'NeuralTextures'),
                                                 fake_test_videos[idx])
        if os.path.isdir(NeuralTextures_video_path):
            if len(os.listdir(NeuralTextures_video_path)):
                temp_save_root_path = '/'
                for name in NeuralTextures_video_path.replace('NeuralTextures',
                                                              'NeuralTextures_ssim_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)
                temp_save_root_path = '/'
                for name in NeuralTextures_video_path.replace('NeuralTextures', 'NeuralTextures_abs_diff').split(
                        '/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)
						
                image_names = os.listdir(NeuralTextures_video_path)
                for image_name in image_names:
                    try:
                        img_pairs = []
                        image_path1 = os.path.join(NeuralTextures_video_path, image_name)
                        image_path2 = os.path.join(real_video_path, image_name)
                        image1, image2 = cv2.imread(image_path1, cv2.IMREAD_COLOR), cv2.imread(image_path2, cv2.IMREAD_COLOR)
                        image1 = image1.astype(np.float32)
                        image2 = image2.astype(np.float32)
                        temp = np.abs(image1 - image2)
                        cv2.imwrite(os.path.join(temp_save_root_path, image_name), temp.astype(np.uint8))
                        d, a = structural_similarity(image1, image2, multichannel=True, data_range=255, full=True)
                        a = 1 - a
                        temp = (a * 255.0).astype(np.uint8)
                        cv2.imwrite(os.path.join(temp_save_root_path.replace('NeuralTextures_abs_diff', 'NeuralTextures_ssim_diff'), image_name), temp)
                    except:
                        continue

        FaceShifter_video_path = os.path.join(image_root_path,
                                              deepfake_sub_path.replace('Deepfakes', 'FaceShifter'),
                                              fake_test_videos[idx])
        if os.path.isdir(FaceShifter_video_path):
            if len(os.listdir(FaceShifter_video_path)):
                temp_save_root_path = '/'
                for name in FaceShifter_video_path.replace('FaceShifter', 'FaceShifter_ssim_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)
                temp_save_root_path = '/'
                for name in FaceShifter_video_path.replace('FaceShifter', 'FaceShifter_abs_diff').split('/'):
                    temp_save_root_path = os.path.join(temp_save_root_path, name)
                    if not os.path.isdir(temp_save_root_path):
                        os.mkdir(temp_save_root_path)

                image_names = os.listdir(FaceShifter_video_path)
                for image_name in image_names:
                    try:
                        img_pairs = []
                        image_path1 = os.path.join(FaceShifter_video_path, image_name)
                        image_path2 = os.path.join(real_video_path, image_name)
                        image1, image2 = cv2.imread(image_path1, cv2.IMREAD_COLOR), cv2.imread(image_path2, cv2.IMREAD_COLOR)
                        image1 = image1.astype(np.float32)
                        image2 = image2.astype(np.float32)
                        temp = np.abs(image1 - image2)
                        cv2.imwrite(os.path.join(temp_save_root_path, image_name), temp.astype(np.uint8))
                        d, a = structural_similarity(image1, image2, multichannel=True, data_range=255, full=True)
                        a = 1 - a
                        temp = (a * 255.0).astype(np.uint8)
                        cv2.imwrite(os.path.join(temp_save_root_path.replace('FaceShifter_abs_diff', 'FaceShifter_ssim_diff'), image_name), temp)
                    except:
                        continue





if __name__ == "__main__":
    image_root_path = '/raid/chenhan/FaceForensics_Fingerprints'
    real_sub_path = 'original_sequences/youtube/c23/videos'
    deepfake_sub_path = 'manipulated_sequences/Deepfakes/c23/videos'

    fake_test_videos = []
    real_test_videos = []

    f = open('splits/train.json', 'r')
    test_json = json.load(f)
    for video_name in test_json:
        fake_test_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_test_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        real_test_videos.append(video_name[0] + '.mp4')
        real_test_videos.append(video_name[1] + '.mp4')    
        
    f = open('splits/val.json', 'r')
    test_json = json.load(f)
    for video_name in test_json:
        fake_test_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_test_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        real_test_videos.append(video_name[0] + '.mp4')
        real_test_videos.append(video_name[1] + '.mp4')
        
    f = open('splits/test.json', 'r')
    test_json = json.load(f)
    for video_name in test_json:
        fake_test_videos.append(video_name[0] + '_' + video_name[1] + '.mp4')
        fake_test_videos.append(video_name[1] + '_' + video_name[0] + '.mp4')
        real_test_videos.append(video_name[0] + '.mp4')
        real_test_videos.append(video_name[1] + '.mp4')  
        

    main(image_root_path, real_sub_path, deepfake_sub_path, real_test_videos, fake_test_videos)
