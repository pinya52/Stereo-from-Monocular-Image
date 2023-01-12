import os
import cv2
from random import shuffle
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
from argparse import ArgumentParser, Namespace
from pathlib import Path
import numpy as np
import struct
import json

from dataset import dispDataset
from model import MiDaSProNet
from loss import ScaleAndShiftInvariantLoss

def average(lst):
    return sum(lst) / len(lst)

def read_pfm(filename):
    with Path(filename).open('rb') as pfm_file:

        line1, line2, line3 = (pfm_file.readline().decode('latin-1').strip() for _ in range(3))
        assert line1 in ('PF', 'Pf')
        
        channels = 3 if "PF" in line1 else 1
        width, height = (int(s) for s in line2.split())
        scale_endianess = float(line3)
        bigendian = scale_endianess > 0
        scale = abs(scale_endianess)

        buffer = pfm_file.read()
        samples = width * height * channels
        assert len(buffer) == samples * 4
        
        fmt = f'{"<>"[bigendian]}{samples}f'
        decoded = struct.unpack(fmt, buffer)
        shape = (height, width, 3) if channels == 3 else (height, width)
        return np.flipud(np.reshape(decoded, shape)) * scale

def loadData():
    seq_data = {'train':[], 'eval':[]}

    print('\n\nReading training data')
    scenes = ['scene_forwards', 'scene_backwards']
    speeds = ['slow', 'fast']
    for scene in scenes:
        for speed  in speeds:
            img_path = './sceneflow/driving__frames_cleanpass/frames_cleanpass/35mm_focallength/%s/%s/left'%(scene, speed)
            disp_path = './sceneflow/driving__disparity/disparity/35mm_focallength/%s/%s/left'%(scene, speed)
            for i, img_name in enumerate(tqdm(sorted([filename for filename in os.listdir(img_path)]))):
                if i == 500:
                    break
                img = cv2.imread('%s/%s'%(img_path, img_name))
                disp = read_pfm('%s/%s'%(disp_path, img_name[:-3]+'pfm'))
                disp = np.tile(disp[..., np.newaxis], 3)

                seq_data['train'].append({'img':img, 'disp':disp})

    seq_data['train'] = dispDataset(seq_data['train'], 'train')
    
    print('\nReading validation data')
    img_path = './kitti/data_scene_flow/training/image_2'
    disp_path = './kitti/data_scene_flow/training/disp_occ_0'
    for img_name in tqdm(sorted([filename for filename in os.listdir(img_path) if filename.endswith("10.png")])):
        img = cv2.imread('%s/%s'%(img_path, img_name))
        disp = cv2.imread('%s/%s'%(disp_path, img_name))

        seq_data['eval'].append({'img':img, 'disp':disp})

    seq_data['eval'] = dispDataset(seq_data['eval'], 'train')

    return seq_data

fig = plt.figure()
y_loss = {'train': [], 'val': []}
x_epoch = []

def draw_curve(current_epoch):
    x_epoch.append(current_epoch)
    plt.plot(x_epoch, y_loss['train'], label='train')
    plt.plot(x_epoch, y_loss['val'], label='val')
    fig.legend()
    fig.savefig(os.path.join('./loss.jpg'))
    plt.clf()

def main():
    seed = 100
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load training and validation data
    print('\nLoad training and validation data')
    datasets = loadData()

    train_dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, collate_fn=datasets['train'].collate_fn)
    eval_datalodaer = torch.utils.data.DataLoader(datasets['eval'], batch_size=args.batch_size, shuffle=True, collate_fn=datasets['eval'].collate_fn)

    # Define our model
    print('\n\nInitializing model')
    model = MiDaSProNet()
    net = model.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = ScaleAndShiftInvariantLoss()

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    print("torch is using GPU : ", 1==torch.cuda.device_count())
    epoch_pbar = trange(args.num_epoch, desc="Epoch")

    best_loss = np.inf
    for epoch in epoch_pbar:
        train_loss = []
        eval_loss = []

        ###################
        # train the model #
        ###################
        net.train()
        for sample in train_dataloader:
            img, disp = sample['img'].to(device), sample['disp'].to(device)
            
            optimizer.zero_grad()
            output = net(img)

            target = torch.stack([disp[i][0] for i in range(disp.size(0))]).to(device)

            mask = torch.full(target.size(), 1).to(device)
            loss = criterion(output, target, mask)
            loss.backward()

            train_loss.append(loss.item())
            optimizer.step()
        
        ######################
        # validate the model #
        ######################
        net.eval()
        for sample in eval_datalodaer:
            img, disp = sample['img'].to(device), sample['disp'].to(device)
            output = net(img)

            target = torch.stack([disp[i][0] for i in range(disp.size(0))]).to(device)

            mask = torch.full(target.size(), 1).to(device)
            loss = criterion(output, target, mask)
            eval_loss.append(loss.item())
            
        avg_train_loss = average(train_loss)
        avg_eval_loss = average(eval_loss)

        y_loss['train'].append(avg_train_loss)
        y_loss['val'].append(avg_eval_loss)
        draw_curve(epoch)

        print('\tEpoch: {:d} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch+1, avg_train_loss, avg_eval_loss))

        if avg_eval_loss < best_loss:
            best_loss = avg_eval_loss
            print('\n\n**********************\n  Found better model  \n**********************\n\n')
            torch.save(model.state_dict(), os.path.join('best.pth'))
        lr_scheduler.step()

    with open("loss.json", "w") as outfile:
        json_object = json.dumps(y_loss, indent = 4)
        outfile.write(json_object)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # data loader
    parser.add_argument("--batch_size", type=int, default=50)

    # training
    parser.add_argument("--num_epoch", type=int, default=500)
    parser.add_argument("--warmup_epoch", type=int, default=10)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print()
    # loadData()
    main()