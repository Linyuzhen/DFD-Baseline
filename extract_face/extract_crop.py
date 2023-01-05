
import torch
import cv2
from facenet_pytorch import MTCNN
from collections import OrderedDict
from PIL import Image
import numpy as np
import retinaface

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def extract_video_halve(input_dir, save_dir, model, scale=1.3):
    reader = cv2.VideoCapture(input_dir)
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 32
    original_frames = OrderedDict()
    halve_frames = OrderedDict()
    for i in range(frames_num):
        reader.grab()
        success, frame = reader.retrieve()

        if not success:
            continue
        original_frames[i] = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize(size=[s // 2 for s in frame.size])
        halve_frames[i] = frame

    original_frames = list(original_frames.values())
    halve_frames = list(halve_frames.values())
    crops = []
    for i in range(0, len(halve_frames), batch_size):
        batch_boxes, batch_probs, batch_points = model.detect(halve_frames[i:i + batch_size], landmarks=True)
        batch_boxes, batch_probs, batch_points = model.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                    halve_frames[i:i + batch_size], method="probability")
        # print(batch_boxes.shape)
        # print(batch_probs.shape)
        # print(batch_points.shape)
        index = -1
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox[0, :]]
                w = xmax - xmin
                h = ymax - ymin
                # p_h = h // 3
                # p_w = w // 3
                size_bb = int(max(w, h) * scale)
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                # Check for out of bounds, x-y top left corner
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                # Check for too big bb size for given x, y
                size_bb = min(original_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(original_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                # crop = original_frames[index][max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                crop = original_frames[i:i + batch_size][index][ymin:ymin+size_bb, xmin:xmin+size_bb]
                crops.append(crop)
            else:
                pass

    for j, crop in enumerate(crops):
        cv2.imwrite(os.path.join(save_dir, "%04d.jpg" % j), crop)


def extract_video(input_dir, save_dir, model, scale=1.3):
    reader = cv2.VideoCapture(input_dir)
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 32
    original_frames = OrderedDict()
    pil_frames = OrderedDict()
    for i in range(frames_num):
        reader.grab()
        success, frame = reader.retrieve()

        if i % 10 == 0:
            if not success:
                continue
            original_frames[i] = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            # frame = frame.resize(size=[s // 2 for s in frame.size])
            pil_frames[i] = frame

    original_frames = list(original_frames.values())
    pil_frames = list(pil_frames.values())
    crops = []
    for i in range(0, len(pil_frames), batch_size):
        batch_boxes, batch_probs, batch_points = model.detect(pil_frames[i:i + batch_size], landmarks=True)
        None_array = np.array([], dtype=np.int16)
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                pass
            else:
                batch_boxes[index] = None_array

        batch_boxes, batch_probs, batch_points = model.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                    pil_frames[i:i + batch_size], method="probability")
        # print(batch_probs.shape)
        # print(batch_points.shape)
        # batch_boxes = np.squeeze(batch_boxes, 1)
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b) for b in bbox[0, :]]
                w = xmax - xmin
                h = ymax - ymin
                # p_h = h // 3
                # p_w = w // 3
                size_bb = int(max(w, h) * scale)
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                # Check for out of bounds, x-y top left corner
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                # Check for too big bb size for given x, y
                size_bb = min(original_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(original_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                # crop = original_frames[index][max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                crop = original_frames[i:i + batch_size][index][ymin:ymin+size_bb, xmin:xmin+size_bb]
                crops.append(crop)
            else:
                pass

    for j, crop in enumerate(crops):
        idx = j*10
        cv2.imwrite(os.path.join(save_dir, "%04d.jpg" % idx), crop)


if __name__ == "__main__":
    input_dir = "/home/biomedia4n6/Public/linyz/datasets/Celeb-DF-v2/video/Celeb-synthesis/id0_id1_0000.mp4"
    save_dir  = "/home/biomedia4n6/Public/linyz/datasets/Celeb-DF-v2/id0_id1_0000.mp4"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    model = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)
    extract_video(input_dir, save_dir, model)


