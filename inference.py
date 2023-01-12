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

from skimage.filters import gaussian, sobel
import copy
from scipy.interpolate import griddata

def xy_to_binary2d(ts):
    '''Convert a list of (x,y) tuples to binary 2d format acceptable to skimage.'''
    if ts.dtype != 'int32': 
        print('Only integer input is supported.')

    xmax,ymax = ts.max(axis=0)
    __,ymin = ts.min(axis=0)

    if ymin < 0:
        print('Negative integers are not supported.')

    r = np.zeros((ymax+2,xmax+2))
    for each in ts:r.itemset(each[1],each[0])

    return r

def loadData(seq):
    seq_data = {'test' : []}

    print('\nReading testing data')
    img_path = './kitti/data_odometry_color/dataset/sequences'
    # seq_list = sorted([filename for filename in os.listdir(img_path)])
    seq_list = [seq]
    for seq in seq_list:
        for img_name in tqdm(sorted([filename for filename in os.listdir('%s/%s/image_2'%(img_path, seq))])):
            path = '%s/%s/image_2/%s'%(img_path, seq, img_name)
            img = cv2.imread(path)
            seq_data['test'].append({'img' : img, 'img_path' : '%s_%s'%(seq, img_name)})
    
    seq_data['test'] = dispDataset(seq_data['test'], 'inference')

    return seq_data

def get_occlusion_mask(shifted, width):

        mask_up = shifted > 0
        mask_down = shifted > 0

        shifted_up = np.ceil(shifted)
        shifted_down = np.floor(shifted)

        for col in range(width - 2):
            loc = shifted[:, col:col + 1]  # keepdims
            loc_up = np.ceil(loc)
            loc_down = np.floor(loc)

            _mask_down = ((shifted_down[:, col + 2:] != loc_down) * (
            (shifted_up[:, col + 2:] != loc_down))).min(-1)
            _mask_up = ((shifted_down[:, col + 2:] != loc_up) * (
            (shifted_up[:, col + 2:] != loc_up))).min(-1)

            mask_up[:, col] = mask_up[:, col] * _mask_up
            mask_down[:, col] = mask_down[:, col] * _mask_down

        mask = mask_up + mask_down
        return mask


def main():
    seed = 100
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load testing data
    print('\nLoad testing data')
    datasets = loadData(args.seq)
    # print(datasets)

    test_dataloader = torch.utils.data.DataLoader(datasets['test'], batch_size=args.batch_size, collate_fn=datasets['test'].collate_fn)

    # Define our model
    print('\n\nInitializing model')
    model = MiDaSProNet()
    model.load_state_dict(torch.load('./wo_range_best.pth'))
    net = model.to(device)

    print("torch is using GPU : ", 1==torch.cuda.device_count())

    net.eval()
    for i, sample in enumerate(tqdm(test_dataloader)):
        img, image, img_name = sample['img'], sample['disp'][1], sample['disp'][0] # disp in test_dataloader is the origin image
        
        inference_img = img.to('cuda')

        max_disp = 0

        with torch.no_grad():
            prediction = net(inference_img)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(image.shape[:2][0], image.shape[:2][1]+max_disp),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            prediction = prediction.cpu().detach().numpy()
            max_disp = prediction.max()

            prediction = abs(prediction / max_disp)

            prediction = np.power(prediction, 3)

            prediction = prediction * 25

            height, width = image.shape[:2][0], image.shape[:2][1]+max_disp

            xs, ys = np.meshgrid(np.arange(width), np.arange(height))

        origin_tensor = image.astype(float)

        origin_tensor = torch.tensor(origin_tensor.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0)
        
        height, width = prediction.shape

        xs, ys = np.meshgrid(range(width), range(height))
        xs, ys = torch.from_numpy(xs).float(), torch.from_numpy(ys).float()

        xs = xs + prediction
        xs = ((xs / (width - 1)) - 0.5) * 2
        ys = ((ys / (height - 1)) - 0.5) * 2
        sample_pix = torch.stack([xs, ys], 2)

        background_image = F.grid_sample(origin_tensor, sample_pix.unsqueeze(0), mode='bilinear',
                                        padding_mode='border', align_corners=True).int()
        background_image = background_image[0].cpu().numpy()
        background_image = background_image.transpose(1,2,0)


        xs, ys = np.meshgrid(np.arange(width), np.arange(height))
        
        warped_image = np.zeros_like(image).astype(float)
        warped_image = np.stack([warped_image] * 2, 0)
        pix_locations = xs - prediction


        """
        filling 
        """
        mask = get_occlusion_mask(pix_locations, width)
        masked_pix_locations = pix_locations * mask - width * (1 - mask)

        weights = np.ones((2, height, width)) * 10000
        for col in range(width - 1, -1, -1):
            loc = masked_pix_locations[:, col]

            loc_up = np.ceil(loc).astype(int)
            loc_down = np.floor(loc).astype(int)
            weight_up = loc_up - loc
            weight_down = 1 - weight_up

            mask = loc_up >= 0
            mask[mask] = \
                weights[0, np.arange(height)[mask], loc_up[mask]] > weight_up[mask]
            weights[0, np.arange(height)[mask], loc_up[mask]] = \
                weight_up[mask]

            warped_image[0, np.arange(height)[mask], loc_up[mask]] = \
                image[:, col][mask] / 255.

            mask = loc_down >= 0
            mask[mask] = \
                weights[1, np.arange(height)[mask], loc_down[mask]] > weight_down[mask]
            weights[1, np.arange(height)[mask], loc_down[mask]] = weight_down[mask]
            warped_image[1, np.arange(height)[mask], loc_down[mask]] = \
                image[:, col][mask] / 255.

        weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
        weights = np.expand_dims(weights, -1)
        warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
        warped_image *= 255.

        warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]

        warped_image = warped_image.astype(np.uint8)
        
        save_path = './warp/%s'%(args.seq)
        if not os.path.exists('./warp'):
            os.mkdir('./warp')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        cv2.imwrite('%s/%s'%(save_path, img_name[4:]), warped_image)


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