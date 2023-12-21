import os
import torch
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from dataset import BCDataset, ToTensorV2
import torch.optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from pytorch_toolbelt import losses as L
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss
from evaluation import get_JS, get_DC, get_F1, get_accuracy, get_sensitivity, get_specificity
from MDFNet import MDFNet, MDFNet_S
import cv2
import torch.nn as nn
import matplotlib.pyplot as plt
from losses.loss import SoftDiceLoss
import csv
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='BUS_A', help='dataset_name') # BUS_A BUS_B
parser.add_argument('--model', type=str, default='MDFNet', help='model_name') # 'MDFNet_S', 'MDFNet'
parser.add_argument('--data_path', type=str, default='./data/', help='path of Dataset')
parser.add_argument('--save_path', type=str, default='./model/', help='path of save')
parser.add_argument('--img_size', type=int, default=320, help='input size')
parser.add_argument('--max_epochs', type=int, default=450, help='maximum epochs to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--total_f', type=int, default=5, help='total folds')
parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
args = parser.parse_args()


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val, size=1):
        metric = self.metrics[metric_name]

        metric["val"] += val*size
        metric["count"] += size
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )



def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    Up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True).to(params["device"])
    Up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True).to(params["device"])
    Up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True).to(params["device"])
    Up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True).to(params["device"])
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True).float().unsqueeze(1)
        size = images.size(0)
        output, x0_c, x1_c, x2_c, x3_c, x4_c = model(images)
        loss = criterion(output, target) + criterion(x0_c, target) + criterion(Up2(x1_c), target) + \
               criterion(Up4(x2_c), target) + criterion(Up8(x3_c), target) + criterion(Up16(x4_c), target)
        dice = get_DC(output.ge(0.5).float(), target)
        metric_monitor.update("Loss", loss.item(), size)
        metric_monitor.update("Dice", dice, size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

        del output, x0_c, x1_c, x2_c, x3_c, x4_c, images, target, loss, dice



def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True).float().unsqueeze(1)
            size = images.size(0)
            output1, x0_c, x1_c, x2_c, x3_c, x4_c = model(images)
            output2, x0_c, x1_c, x2_c, x3_c, x4_c = model(torch.flip(images, [0, 3]))
            output2 = torch.flip(output2, [3, 0])
            output3, x0_c, x1_c, x2_c, x3_c, x4_c = model(torch.flip(images, [0, 2]))
            output3 = torch.flip(output3, [2, 0])
            output = (output1 + output2 + output3) / 3.0
            loss = criterion(output, target)
            output = output.ge(0.5).float()
            dice = get_DC(output, target)
            metric_monitor.update("Loss", loss.item(), size)
            metric_monitor.update("Dice", dice, size)
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )


            del output, x0_c, x1_c, x2_c, x3_c, x4_c, images, target, loss, dice

    dice = metric_monitor.metrics['Dice']['avg']
    return dice


def test(val_loader, model, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    Iou = smp.utils.metrics.IoU().to(params["device"])
    ACC = smp.utils.metrics.Accuracy().to(params["device"])
    csv_root = params['model_save_root'].replace('.pth', '_score.csv')
    print(csv_root)
    fo = open(csv_root, 'w')
    f_csv = csv.writer(fo)
    headers = ['ID','Dice','Iou','','']
    f_csv.writerow(headers)
    with torch.no_grad():
        for i, (images, target, label) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True).float().unsqueeze(1)
            size = images.size(0)
            output1, x0_c, x1_c, x2_c, x3_c, x4_c = model(images)
            x0_c1 = x0_c
            output2, x0_c, x1_c, x2_c, x3_c, x4_c = model(torch.flip(images, [0, 3]))
            output2 = torch.flip(output2, [3, 0])
            x0_c2 = torch.flip(x0_c, [3, 0])
            output3, x0_c, x1_c, x2_c, x3_c, x4_c = model(torch.flip(images, [0, 2]))
            output3 = torch.flip(output3, [2, 0])
            x0_c3 = torch.flip(x0_c, [2, 0])
            output = (output1 + output2 + output3) / 3.0
            

            predicted_masks = output.ge(0.5).float()


            acc = get_accuracy(predicted_masks, target)
            iou = Iou(predicted_masks, target)
            dice = get_DC(predicted_masks, target)
            sensitivity = get_sensitivity(predicted_masks, target)
            specificity = get_specificity(predicted_masks, target)
            metric_monitor.update("Dice", dice, size)
            metric_monitor.update("IOU", iou.item(), size)
            metric_monitor.update("Acc", acc, size)
            metric_monitor.update("sensitivity", sensitivity, size)
            metric_monitor.update("specificity", specificity, size)
            print(i, dice, iou.item(), acc, sensitivity, specificity)
            row = [i, dice, iou.item()]
            f_csv.writerow(row)
            if label[0] == 0:
                metric_monitor.update("bDice", dice, size)
                metric_monitor.update("bIOU", iou.item(), size)
                metric_monitor.update("bAcc", acc, size)
                metric_monitor.update("bsensitivity", sensitivity, size)
                metric_monitor.update("bspecificity", specificity, size)
            else:
                metric_monitor.update("mDice", dice, size)
                metric_monitor.update("mIOU", iou.item(), size)
                metric_monitor.update("mAcc", acc, size)
                metric_monitor.update("msensitivity", sensitivity, size)
                metric_monitor.update("mspecificity", specificity, size)


    dice = metric_monitor.metrics['Dice']['avg']
    iou = metric_monitor.metrics['IOU']['avg']
    acc = metric_monitor.metrics['Acc']['avg']
    sensitivity = metric_monitor.metrics['sensitivity']['avg']
    specificity = metric_monitor.metrics['specificity']['avg']


    bdice = metric_monitor.metrics['bDice']['avg']
    biou = metric_monitor.metrics['bIOU']['avg']
    bsensitivity = metric_monitor.metrics['bsensitivity']['avg']
    bspecificity = metric_monitor.metrics['bspecificity']['avg']

    mdice = metric_monitor.metrics['mDice']['avg']
    miou = metric_monitor.metrics['mIOU']['avg']
    msensitivity = metric_monitor.metrics['msensitivity']['avg']
    mspecificity = metric_monitor.metrics['mspecificity']['avg']
    print('b: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(bdice, biou, bsensitivity, bspecificity))
    print('m: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(mdice, miou, msensitivity, mspecificity))
    print('a: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format( dice,  iou,  sensitivity,  specificity))

    rows = [
    ['Type', 'dice', 'iou', 'se', 'sp'],
    ['Benign', bdice, biou, bsensitivity, bspecificity],
    ['Malignant', mdice, miou, msensitivity, mspecificity],
    ['All', dice, iou, sensitivity, specificity]]
    f_csv.writerows(rows)
    
    fo.close()
    return dice, iou, acc, sensitivity, specificity


def train_and_validate(model, train_dataset, val_dataset, params):
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params["num_workers"],
        drop_last=False,
        pin_memory=True,
    )

    DiceLoss_fn=SoftDiceLoss()
    BCE_fn=nn.BCEWithLogitsLoss()
    criterion = L.JointLoss(first=DiceLoss_fn, second=BCE_fn, first_weight=0.5, second_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params['weight_decay'])
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=params["milestones"], gamma=params["gamma"])
    max_dice = 0
    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        dice = validate(val_loader, model, criterion, epoch, params)
        if dice <= max_dice:
            print('Epoch: {}, lr: {:.6f},  Dice: {:.4f}'.format(epoch, exp_lr_scheduler.get_last_lr()[0], max_dice))
        else:
            max_dice = dice
            print('Epoch: {}, lr: {:.6f},  Dice: {:.4f}, model saved!'.format(epoch, exp_lr_scheduler.get_last_lr()[0], max_dice))
            torch.save(model, params["model_save_root"])
        exp_lr_scheduler.step()
        del dice
    print('Finished! Max dice: ', max_dice, '\n')

def create_model(params):
    model = params["model"]
    model = model.to(params["device"])
    return model
    
    
if __name__ == "__main__":
    
    img_size = args.img_size
    net = args.model
    dataset = args.dataset_name
    if dataset == "BUS_A":
        mean = 0.1237
        std = 0.1462
    elif dataset == "BUS_B":
        mean = 0.1648
        std = 0.1957   
    
    train_transform1 = A.Compose(
        [
            A.Resize(int(img_size*1.1), int(img_size*1.1)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomCrop(img_size, img_size),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ToTensorV2(),
        ]
    )
    
    train_transform2 = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ToTensorV2(),
        ]
    )
    
    val_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ToTensorV2(),
        ]
    )
    
    for i in range(args.total_f):
        
        n = i + 1
        train_lst = args.data_path + dataset + '/train'+str(n)+'.txt'
        val_lst = args.data_path + dataset + '/val'+str(n)+'.txt'
        model_root = args.save_path + dataset + '_' + net + '_f' + str(n) + '.pth'
        train_dataset = BCDataset(args.data_path, train_lst, transform=train_transform1, transform2=train_transform2)
        val_dataset = BCDataset(args.data_path, val_lst, transform=val_transform)
        net_dict = {
            'MDFNet': MDFNet(out_ch=1),
            'MDFNet_S': MDFNet_S(out_ch=1)
        }
        params = {
            "model": net_dict[net],
            "device": "cuda",
            "lr": 0.001,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "epochs": args.max_epochs,
            "milestones": [i for i in range(15, args.max_epochs, 15)],
            "model_save_root": model_root,
            'gamma': 0.9,
            'weight_decay': 5e-4,
        }
        print('{} {} fold {}'.format(dataset, net, n))
        model = create_model(params)
        train_and_validate(model, train_dataset, val_dataset, params)


        val_dataset = BCDataset(args.data_path, val_lst, transform=val_transform, inf=True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )          
        model = torch.load(params['model_save_root'])
        dice, iou, acc, sensitivity, specificity = test(val_loader, model, params)
        print('Dataset: {}, Model: {}, Dice: {:.4f}, IoU: {:.4f}, ACC: {:.4f}, Sen: {:.4f}, Spe: {:.4f}'.format(dataset, net, dice, iou, acc, sensitivity, specificity))