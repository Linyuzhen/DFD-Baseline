import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse

import numpy as np
from datasets.transform import build_transforms
from networks.models import model_selection
from utils.metrics import get_metrics
from datasets.DFD_datasets import FFDataset, CDFDataset, SimpleDatasets


def test_model(model, dataloaders):
    prediction = np.array([])
    model.train(False)
    model.eval()
    for images in dataloaders:
        input_images = Variable(images.cuda())
        outputs = model(input_images)
        pred_ = torch.nn.functional.softmax(outputs, dim=-1)
        pred_ = pred_.cpu().detach().numpy()[:, 1]

        prediction = np.insert(prediction, 0, pred_)
    return prediction


def parse_args():
    parser = argparse.ArgumentParser(description='Testing network on FaceForensics and CelebDF')

    parser.add_argument('--root_path', type=str, default='/home/biomedia4n6/Public/linyz/datasets/FF-face')
    parser.add_argument('--cdf_path', type=str, default='/home/biomedia4n6/Public/linyz/datasets/CDF-face')
    parser.add_argument('--save_path', type=str, default='./save_result')
    parser.add_argument('--save_test_path', type=str, default='./test_results')

    parser.add_argument('--backbone', type=str, default='efn-b4')
    parser.add_argument('--train_method', type=str, default='Deepfakes')
    parser.add_argument('--test_method', type=str, default='CDF')
    parser.add_argument('--qp', type=str, default='c40')

    parser.add_argument('--gpu_id', type=int, default=2)
    parser.add_argument('--continue_train', type=bool, default=False)

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_name', type=list,
                        default=['real', 'fake'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--img_size', type=int, default=380)
    parser.add_argument('--test_frame_nums', type=int, default=5)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    result_txt = open('{}/{}_{}_{}_test.txt'.format(args.save_test_path, args.qp,
                                                    args.train_method, args.test_method)
                      , 'w', encoding='utf-8')

    model_path = os.path.join(args.save_path, args.backbone, 'models')
    data_transforms = build_transforms(height=args.img_size, width=args.img_size)

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

    # create model
    model_name = '{}_{}'.format(args.train_method, args.qp)
    print("=> creating pretrained model {} on {}".format(args.backbone, model_name))

    model, image_size, *_ = model_selection(modelname=args.backbone, num_out_classes=2, dropout=0.5)
    model = nn.DataParallel(model)

    # load pretrained model
    saved_model = os.path.join(model_path, '{}_{}_best.pth'.format(args.train_method, args.qp))
    model.load_state_dict(torch.load(saved_model))
    model = model.cuda()

    image_prediction = np.array([])
    image_label = np.array([])
    video_prediction = np.zeros(len(test_videos), dtype=np.float32)
    video_label = np.zeros(len(test_videos), dtype=np.float32)

    for idx, video_name in enumerate(test_videos):
        video_path = os.path.join(args.root_path, video_name)

        test_dataset = SimpleDatasets(video_path, transform=data_transforms)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                  drop_last=False, shuffle=False, num_workers=4, pin_memory=True)

        with torch.no_grad():
            prediction = test_model(model, test_loader)

        if len(prediction) != 0:
            video_prediction[idx] = np.mean(prediction, axis=0)
        else:
            video_prediction[idx] = 0.5  # default is 0.5
        # print(video_name + '  is  fake' if np.round(video_prediction[idx]) == 1
        #       else video_name + '  is  real')
        # print('Real %f' % video_prediction[idx])

        if (video_name.find('original') != -1) or (video_name.find('real') != -1):
            label = np.zeros(prediction.shape, dtype=np.float32)
            video_label[idx] = 0.0
        else:
            label = np.ones(prediction.shape, dtype=np.float32)
            video_label[idx] = 1.0

        image_prediction = np.insert(image_prediction, 0, prediction)
        image_label = np.insert(image_label, 0, label)
        result_txt.write(video_name + 'real_probability %d' % np.round(video_prediction[idx]) + '\n')

    recall, precision, auc, EER, f1, accuracy = get_metrics(image_label, image_prediction)
    print('Image Result recall: {:g} precision: {:g} F1: {:g} AUC: {:g} EER: {:g} ACC: {:g}  '.
          format(recall, precision, f1, auc, EER, accuracy))
    result_txt.write('Image Result recall: {:g} precision: {:g} F1: {:g} AUC: {:g} EER: {:g} ACC: {:g}  \n'.
                     format(recall, precision, f1, auc, EER, accuracy))

    recall, precision, auc, EER, f1, accuracy = get_metrics(video_label, video_prediction)
    print('Video Result recall: {:g} precision: {:g} F1: {:g} AUC: {:g} EER: {:g} ACC: {:g}  '.
          format(recall, precision, f1, auc, EER, accuracy))
    result_txt.write('Video Result recall: {:g} precision: {:g} F1: {:g} AUC: {:g} EER: {:g} ACC: {:g}  \n'.
                     format(recall, precision, f1, auc, EER, accuracy))
    result_txt.write('{} Model, Training on {}_{}, Testing on {}'.
                     format(args.backbone, args.qp, args.train_method, args.test_method))


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # save_test_path = os.path.join(args.save_test_path, args.backbone)
    if not os.path.exists(args.save_test_path):
        os.makedirs(args.save_test_path)
    main()
