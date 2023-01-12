from typing import List, Dict

from torch.utils.data import Dataset

import torch


class dispDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        task = 'train'
    ):
        self.data = data
        self.task = task
        model_type = 'MiDaS_small'
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    def collate_fn(self, samples: List[Dict]) -> Dict:
        img_list = []
        disp_list = []

        for sample in samples:
            # don't have groundtruth for inference, use disp_list to store original image
            if self.task == 'train':
                disp_list.append(self.transform(sample['disp']))
                print(disp_list[-1].shape)
            else:
                disp_list.append(sample['img_path'])
                disp_list.append(sample['img'])

            img_list.append(self.transform(sample['img']))

        img_list = torch.stack(img_list).squeeze(1)

        if self.task == 'train':
            disp_list = torch.stack(disp_list).squeeze(1)

        return {'img' : img_list, 'disp' : disp_list}
