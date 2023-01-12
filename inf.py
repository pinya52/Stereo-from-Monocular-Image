import os
import cv2
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange, tqdm
from argparse import ArgumentParser, Namespace

from dataset import dispDataset
from model import MiDaSProNet

def loadData(seq):
    seq_data = {'test' : []}

    print('\nReading testing data')
    img_path = './kitti/data_odometry_color/dataset/sequences'
    seq_list = [seq]
    for seq in seq_list:
        for img_name in tqdm(sorted([filename for filename in os.listdir('%s/%s/image_2'%(img_path, seq))])):
            path = '%s/%s/image_2/%s'%(img_path, seq, img_name)
            img = cv2.imread(path)
            seq_data['test'].append({'img' : img, 'img_path' : '%s_%s'%(seq, img_name)})
    
    seq_data['test'] = dispDataset(seq_data['test'], 'inference')

    return seq_data

def main():
    seed = 100
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load testing data
    print('\nLoad testing data')
    datasets = loadData(args.seq)

    test_dataloader = torch.utils.data.DataLoader(datasets['test'], batch_size=args.batch_size, collate_fn=datasets['test'].collate_fn)

    # Define our model
    print('\n\nInitializing model')
    model = MiDaSProNet()
    model.load_state_dict(torch.load('./wo_range_best.pth'))
    net = model.to(device)

    print("torch is using GPU : ", 1==torch.cuda.device_count())

    net.eval()
    for i, sample in enumerate(tqdm(test_dataloader)):
        if i == 0:
            background_image = sample['disp'][1]
            continue
        img, origin, img_name = sample['img'], sample['disp'][1], sample['disp'][0] # disp in test_dataloader is the origin image
        inference_img = img.to('cuda')

        with torch.no_grad():
            prediction = net(inference_img)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=origin.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        prediction = prediction.cpu().numpy()
        origin_tensor = origin.astype(float)

        origin_tensor = torch.tensor(origin_tensor.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0)
        
        height, width = prediction.shape

        xs, ys = np.meshgrid(range(width), range(height))
        xs, ys = torch.from_numpy(xs).float(), torch.from_numpy(ys).float()

        xs = xs + prediction
        xs = ((xs / (width - 1)) - 0.5) * 2
        ys = ((ys / (height - 1)) - 0.5) * 2
        sample_pix = torch.stack([xs, ys], 2)

        warped_image = F.grid_sample(origin_tensor, sample_pix.unsqueeze(0), mode='bilinear',
                                        padding_mode='border', align_corners=True).int()
        warped_image = warped_image[0].cpu().numpy()
        warped_image = warped_image.transpose(1,2,0)

        os.mkdir('./warp_img/warp/%s'%(args.seq))
        cv2.imwrite('./warp_img/warp/%s/%s.png'%(args.seq, img_name), warped_image)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    # data loader
    parser.add_argument("--batch_size", type=int, default=1)

    # training
    parser.add_argument("--num_epoch", type=int, default=200)
    
    parser.add_argument("--seq", type=str, default=00)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print()
    # loadData()
    main()