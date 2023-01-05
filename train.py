import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import argparse

import numpy as np
import pandas as pd
from datasets.transform import build_transforms
from networks.models import model_selection
from utils.metrics import get_metrics
from datasets.DFD_datasets import FFDataset


def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network


def parse_args():
    parser = argparse.ArgumentParser(description='Training network on FaceForensics')

    parser.add_argument('--root_path', type=str, default='/home/biomedia4n6/Public/linyz/datasets/FF-face')
    parser.add_argument('--save_path', type=str, default='./save_result')

    parser.add_argument('--backbone', type=str, default='efn-b4')
    parser.add_argument('--method', type=str, default='Deepfakes')
    parser.add_argument('--qp', type=str, default='c23')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--continue_train', type=bool, default=False)
    parser.add_argument('--model_path', '-mp', type=str, default='Pre_model')

    parser.add_argument('--num_class', type=int, default=2)
    parser.add_argument('--class_name', type=list,
                        default=['real', 'fake'])
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--val_epochs', type=int, default=2)
    parser.add_argument('--adjust_lr_epochs', type=int, default=2)
    parser.add_argument('--base_lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=380)
    parser.add_argument('--test_frame_nums', type=int, default=5)
    parser.add_argument('--val_batch_size', type=int, default=256)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    save_path = os.path.join(args.save_path, args.backbone)
    data_transforms = build_transforms(height=args.img_size, width=args.img_size)

    with open('save_txt/train_ff.txt', 'r') as f:
        train_videos = f.readlines()
        train_videos = [
            os.path.join(args.root_path, i.strip().replace('Deepfakes', args.method).replace('c23', args.qp))
            for i in train_videos]
    with open('save_txt/val_ff.txt', 'r') as f:
        val_videos = f.readlines()
        val_videos = [os.path.join(args.root_path, i.strip().replace('Deepfakes', args.method).replace('c23', args.qp))
                      for i in val_videos]

    # TODO:FF all data
    # if args.method == 'all':
    #     pass FFDatasets_all


    train_dataset = FFDataset(video_names=train_videos, phase='train',
                              is_pillow=True, transform=data_transforms)
    val_dataset = FFDataset(video_names=val_videos, phase='valid',
                            is_pillow=True, transform=data_transforms)
    # test_dataset = FFDataset(video_names=test_videos, phase='test',
    #                          is_pillow=False,transform=test_transform)

    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False,
                                               shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=args.val_batch_size,
                                             drop_last=False, num_workers=4, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.val_batch_size,
    #                                           drop_last=True, num_workers=4, pin_memory=True)

    print('All Train videos Number: %d' % (
            len(train_dataset.videos_by_class['real']) + len(train_dataset.videos_by_class['fake'])))

    # create model
    model_name = '{}_{}'.format(args.method, args.qp)
    print("=> creating model {} for {}".format(args.backbone, model_name))

    # model, image_size, *_ = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
    model, image_size, *_ = model_selection(modelname=args.backbone, num_out_classes=2, dropout=0.5)

    # load saved model
    if args.continue_train:
        model.load_state_dict(torch.load(args.model_path))
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    model = nn.DataParallel(model)
    iteration = 0
    best_acc = 0.0

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0
        train_corrects = 0.0

        # train stage
        model.train()
        for images, labels in train_loader:
            # wrap them in Variable
            images = Variable(images.cuda().detach())
            labels = Variable(labels.cuda().detach())

            optimizer.zero_grad()
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            iter_loss = loss.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            train_corrects += iter_corrects
            iteration += 1

        epoch_loss = train_loss * args.batch_size / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        scheduler.step()

        # val stage
        if (epoch + 1) % args.val_epochs == 0:
            model.eval()

            y_true = np.array([])
            y_pred = np.array([])
            correct = {class_index: 0 for class_index in range(args.num_class)}
            total = {class_index: 0 for class_index in range(args.num_class)}
            total_val_loss = 0.0

            for images, labels in val_loader:
                # wrap them in Variable
                y_true = np.insert(y_true, 0, labels.numpy())

                images = Variable(images.cuda().detach())
                labels = Variable(labels.cuda().detach())

                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.no_grad():
                    outputs = model(images)
                # Calculate loss
                loss = criterion(outputs, labels)

                prediction = nn.functional.softmax(outputs, dim=-1)
                y_pred = np.insert(y_pred, 0, prediction.cpu().detach().numpy()[:, 1])
                prediction = torch.argmax(prediction, dim=-1)
                res = prediction == labels
                for label_idx in range(len(labels)):
                    label_single = labels[label_idx].item()
                    correct[label_single] += res[label_idx].item()
                    total[label_single] += 1

                # statistics loss
                iter_loss = loss.item()
                total_val_loss += iter_loss

            val_loss = total_val_loss * args.val_batch_size  / train_dataset_size

            recall, precision, auc, EER, f1, accuracy = get_metrics(y_true, y_pred)

            df_acc = pd.DataFrame()
            df_acc['epoch'] = [epoch]
            df_acc['BCE'] = [val_loss / len(val_loader)]
            df_acc['recall'] = [recall]
            df_acc['precision'] = [precision]
            df_acc['auc'] = [auc]
            df_acc['EER'] = [EER]
            df_acc['f1'] = [f1]
            df_acc['accuracy'] = [accuracy]

            sum_correct = 0
            sum_total = 0
            for idx in range(args.num_class):
                sum_correct += correct[idx]
                sum_total += total[idx]
                df_acc[args.class_name[idx]] = correct[idx] / total[idx]
            avg_acc = sum_correct / sum_total
            df_acc['Acc'] = avg_acc
            if epoch + 1 != (args.val_epochs):
                df_acc.to_csv('%s/report/%s_val.csv' % (save_path, model_name), mode='a',
                              index=None, header=None)
            else:
                df_acc.to_csv('%s/report/%s_val.csv' % (save_path, model_name), mode='a',
                              index=None)

            print('Epoch: {:g}, Val_loss: {:4f}, ACC: {:4f}, AUC: {:4f}'.
                  format(epoch, val_loss / len(val_loader), avg_acc, auc))

            if best_acc < avg_acc:
                best_acc = avg_acc
                save_network(model, '%s/models/%s_best.pth' % (save_path, model_name))


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    save_path = os.path.join(args.save_path, args.backbone)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists('%s/models' % save_path):
        os.makedirs('%s/models' % save_path)
    if not os.path.exists('%s/report' % save_path):
        os.makedirs('%s/report' % save_path)
    # if not os.path.exists('%s/images' % args.save_path):
    #     os.makedirs('%s/images' % args.save_path)
    main()
