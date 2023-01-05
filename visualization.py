import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np

from random import choice
import torchvision
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from datasets.transform import build_transforms
from networks.models import model_selection
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from datasets.DFD_datasets import FFDataset, CDFDataset, SimpleDatasets


def parse_args():
    parser = argparse.ArgumentParser(description='Testing network on FaceForensics and CelebDF')

    parser.add_argument('--root_path', type=str, default='/home/biomedia4n6/Public/linyz/datasets/FF-face')
    parser.add_argument('--cdf_path', type=str, default='/home/biomedia4n6/Public/linyz/datasets/CDF-face')
    parser.add_argument('--save_path', type=str, default='./save_result')
    # parser.add_argument('--save_test_path', type=str, default='./test_results')
    # parser.add_argument('--img_path', type=str, default='vis_imgs')

    parser.add_argument('--backbone', type=str, default='xception')
    parser.add_argument('--train_method', type=str, default='Deepfakes')
    parser.add_argument('--test_method', type=str, default='Deepfakes')
    parser.add_argument('--qp', type=str, default='c23')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=299)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    model_path = os.path.join(args.save_path, args.backbone, 'models')

    # random choose and load image
    if args.test_method == 'CDF':
        txt_name = 'List_of_testing_videos.txt'

        with open(os.path.join(args.cdf_path, txt_name), 'r') as f:
            test_videos = f.readlines()
            test_videos = [os.path.join(args.cdf_path, i.strip().split(' ')[1]) for i in test_videos]
    else:
        with open('save_txt/test_ff.txt', 'r') as f:
            test_videos = f.readlines()
            test_videos = [
                os.path.join(args.root_path, i.strip().replace('Deepfakes', args.test_method).replace('c23', args.qp))
                for i in test_videos]

    test_video = choice(test_videos)
    test_image = choice(os.listdir(test_video))
    print("test image from: {}_{}".format(test_video, test_image))

    img = read_image(os.path.join(test_video, test_image))
    input_tensor = normalize(resize(img, (args.img_size, args.img_size)) / 255., [0.5] * 3, [0.5] * 3)

    # input_tensor = SimpleDatasets(os.path.join(test_video, test_image))

    # create model
    model_name = '{}_{}'.format(args.train_method, args.qp)
    print("=> creating pretrained model {} on {}".format(args.backbone, model_name))

    model, image_size, *_ = model_selection(modelname=args.backbone, num_out_classes=2, dropout=0.5)
    model = nn.DataParallel(model)

    # load pretrained model
    saved_model = os.path.join(model_path, '{}_{}_best.pth'.format(args.train_method, args.qp))
    model.load_state_dict(torch.load(saved_model))
    model.eval()
    # for name, param in model.named_parameters():
    #     print(name)
    # model = model.cuda()

    # cam_extractor = SmoothGradCAMpp(model, 'module.model.layer4')
    cam_extractor = SmoothGradCAMpp(model, 'module.model.conv4')
    scores = model(input_tensor.unsqueeze(0))
    print(scores)
    print(scores.squeeze(0).argmax().item())
    # cam = cam_extractor(class_idx=1,scores=scores)
    activation_map = cam_extractor(scores.squeeze(0).argmax().item(), scores)
    print(activation_map[0].shape)

    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)

    plt.imshow(to_pil_image(img))
    plt.show()
    plt.imshow(activation_map[0].squeeze(0).cpu().numpy())
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.imshow(result)
    plt.show()




if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # save_test_path = os.path.join(args.save_test_path, args.backbone)
    # if not os.path.exists(args.save_test_path):
    #     os.makedirs(args.save_test_path)
    main()
