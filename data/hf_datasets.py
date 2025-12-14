import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from IMDLBenCo.transforms import get_albu_transforms, EdgeMaskGenerator
from IMDLBenCo.registry import DATASETS

import random


def pil_loader(path: str) -> Image.Image:
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


@DATASETS.register_module()
class HFCLIPSAMDataset(Dataset):
    def __init__(self, path, split_file, split_name, pixel=False,
                 is_padding=False,
                 is_resizing=False,
                 output_size=(1024, 1024),
                 post_transform=None,
                 post_transform_sam=None,
                 common_transforms=None,
                 edge_width=None,
                 img_loader=pil_loader,
                 ) -> None:
        super().__init__()
        self.dataset = load_dataset(path)[split_name]
        with open(split_file, 'r') as f:
            train_indices = json.load(f)
        self.dataset = self.dataset.select(train_indices)
        if pixel:
            filtered_indices = [
                i for i, key in enumerate(self.dataset['key'])
                if ('partial' in key and 'fake' in key) or (path== "nebula/tmpdata2" and 'fake' in key)
            ]
            self.dataset = self.dataset.select(filtered_indices)


        self.output_size = output_size
        self.common_transforms = common_transforms
        self.post_transform = post_transform
        self.post_transform_sam = post_transform_sam
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)
        self.img_loader = img_loader 
        self.is_padding = is_padding
        self.is_resizing = is_resizing


    def _prepare_gt_img(self, tp_img, gt_path, label):
        if label == 0 and gt_path is None:
            return np.zeros((*tp_img.shape[:2], 3))
        elif label == 1 and gt_path is None:
            return np.full((*tp_img.shape[:2], 3), 255, dtype=np.uint8)
        else:
            return np.array(gt_path.convert('RGB'))



    def _process_masks(self, gt_img):
        gt_img = (np.mean(gt_img, axis=2, keepdims=True) > 127.5) * 1.0
        gt_img = gt_img.transpose(2, 0, 1)[0]
        masks_list = [gt_img]
        if self.edge_mask_generator:
            gt_img_edge = self.edge_mask_generator(gt_img)[0][0]
            masks_list.append(gt_img_edge)
        return masks_list

    def __getitem__(self, index):
        sample = self.dataset[index]
        tp_img = np.array(sample['image'].convert('RGB'))
        gt_img = self._prepare_gt_img(tp_img, sample['mask'], sample['label'])

        if self.common_transforms:
            res_dict = self.common_transforms(image=tp_img, mask=gt_img)
            tp_img, gt_img = res_dict['image'], res_dict['mask']
            label = 0 if np.all(gt_img == 0) else 1
        else:
            label = sample['label']

        masks_list = self._process_masks(gt_img)
        res_dict = self.post_transform(image=tp_img, masks=masks_list)
        res_dict_sam = self.post_transform_sam(image=tp_img, masks=masks_list)
        data_dict = {
            'image': res_dict['image'],
            'mask': res_dict['masks'][0].unsqueeze(0),
            'image_sam': res_dict_sam['image'],
            'mask_sam': res_dict_sam['masks'][0].unsqueeze(0),
            'label': label,
            'shape': torch.tensor(self.output_size if self.is_resizing else tp_img.shape[:2]),
            'name': sample['key']
        }

        if self.edge_mask_generator:
            data_dict['edge_mask'] = res_dict['masks'][1].unsqueeze(0)

        if self.is_padding:
            shape_mask = torch.zeros_like(data_dict['mask'])
            shape_mask[:, :data_dict['shape'][0], :data_dict['shape'][1]] = 1
            data_dict['shape_mask'] = shape_mask
        del sample
        return data_dict

    def __len__(self):
        return len(self.dataset)


from IMDLBenCo.model_zoo.cat_net.cat_net_post_function import cat_net_post_func



@DATASETS.register_module()
class HFDataset(Dataset):
    def __init__(self, path, split_name,  pixel=False,
                 is_padding=False,
                 is_resizing=False,
                 output_size=(1024, 1024),
                 post_transform=None,
                 post_transform_sam=None,
                 common_transforms=None,
                 edge_width=None,
                 img_loader=pil_loader,
                 ) -> None:
        super().__init__()
        self.dataset = load_dataset(path)[split_name]
        if pixel:
            filtered_indices = [
                i for i, key in enumerate(self.dataset['key'])
                if ('partial' in key and 'fake' in key) or (path== "nebula/tmpdata2" and 'fake' in key) or (path== "nebula/tempdata4" and 'fake' in key)
            ]
            self.dataset = self.dataset.select(filtered_indices)

        self.output_size = output_size
        self.common_transforms = common_transforms
        self.post_transform = post_transform
        self.post_transform_sam = post_transform_sam
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)
        self.img_loader = img_loader
        self.is_padding = is_padding
        self.is_resizing = is_resizing
        if post_transform is None:
            self.post_transform = get_albu_transforms(type_="pad"
            if is_padding else "resize", output_size=output_size)

    def _prepare_gt_img(self, tp_img, gt_path, label):
        if label == 0:
            return np.zeros((*tp_img.shape[:2], 3))
        elif label == 1 and gt_path is None:
            return np.full((*tp_img.shape[:2], 3), 255, dtype=np.uint8)
        else:
            return np.array(gt_path.convert('RGB'))


    def _process_masks(self, gt_img):
        gt_img = (np.mean(gt_img, axis=2, keepdims=True) > 127.5) * 1.0
        gt_img = gt_img.transpose(2, 0, 1)[0]
        masks_list = [gt_img]
        if self.edge_mask_generator:
            gt_img_edge = self.edge_mask_generator(gt_img)[0][0]
            masks_list.append(gt_img_edge)
        return masks_list

    def __getitem__(self, index):
        sample = self.dataset[index]
        tp_img = np.array(sample['image'].convert('RGB'))
        gt_img = self._prepare_gt_img(tp_img, sample['mask'], sample['label'])

        if self.common_transforms:
            res_dict = self.common_transforms(image=tp_img, mask=gt_img)
            tp_img, gt_img = res_dict['image'], res_dict['mask']
            # import pdb;pdb.set_trace()
            if np.sum(gt_img>0):
                label=1
            else:
                label=0
            # label = sample['label']
        else:
            label = sample['label']
        # label = sample['label']

        # if label != sample['label']:
        #     print(sample['key'])
        #     print(sample['label'])
        #     print(label)

        masks_list = self._process_masks(gt_img)
        res_dict = self.post_transform(image=tp_img, masks=masks_list)

        data_dict = {
            'image': res_dict['image'],
            'mask': res_dict['masks'][0].unsqueeze(0),
            'label': label,
            'shape': torch.tensor(self.output_size if self.is_resizing else tp_img.shape[:2]),
            'name': sample['key']
        }

        if self.edge_mask_generator:
            data_dict['edge_mask'] = res_dict['masks'][1].unsqueeze(0)

        if self.is_padding:
            shape_mask = torch.zeros_like(data_dict['mask'])
            shape_mask[:, :data_dict['shape'][0], :data_dict['shape'][1]] = 1
            data_dict['shape_mask'] = shape_mask

        cat_net_post_func(data_dict)
        del sample
        del res_dict

        return data_dict

    def __len__(self):
        return len(self.dataset)


# @DATASETS.register_module()
class BalancedDataset(Dataset):
    def __init__(self, path, split_name,  pixel=False,
                 is_padding=False,
                 is_resizing=False,
                 output_size=(1024, 1024),
                 post_transform=None,
                 post_transform_sam=None,
                 common_transforms=None,
                 edge_width=None,
                 img_loader=pil_loader,
                 sample_number=1840
                 ) -> None:
        super().__init__()
        
        self.split = split_name
        self.sampler_number = sample_number
        
        self.output_size = output_size
        self.common_transforms = common_transforms
        self.post_transform = post_transform
        self.post_transform_sam = post_transform_sam
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)
        self.img_loader = img_loader
        self.is_padding = is_padding
        self.is_resizing = is_resizing
        if post_transform is None:
            self.post_transform = get_albu_transforms(type_="pad"
            if is_padding else "resize", output_size=output_size)
        
        self.dataset=[]
        self.dataset_border=[0,]
        # 这里改成加载json文件中的路径，然后给定对应的cls_labels。整体文件顺序按照数据集排序
        with open(path, "r") as f:
            datasets = json.load(f)
        print(f"[DEBUG] datasets path: {datasets}")
        # datasets_path = [json_path for _,json_path in datasets]
        dataset_names = []
        dataset_jsons = []
        for arr in datasets:
            dataset_names.append(arr[0])
            dataset_jsons.append(arr[1])
        
        self.num_datasets = len(datasets)
        
        for i in range(self.num_datasets):
            data_name = dataset_names[i]
            data_path = dataset_jsons[i]
            print(f"[DEBUG] [{i+1}/{self.num_datasets}] loading {data_name}, path is {data_path}")
            with open(data_path, "r") as f:
                dataset = json.load(f)
            for pair in dataset:
                img_path = pair[0]
                mask_path = pair[1]
                # print(f"[DEBUG]: img_path = {img_path}, mask_path = {mask_path}")
                if os.path.isfile(img_path):
                    data_sample ={'key':None, 'image': None, 'mask': None, 'label': 0 }
                    data_sample['key'] = img_path
                    data_sample['image'] = img_path
                    data_sample['mask'] = mask_path if mask_path != "Negative" else None
                    data_sample['label'] = 0 if mask_path == "Negative" else 1
                    self.dataset.append(data_sample)
            self.dataset_border.append(len(self.dataset))

    def _prepare_gt_img(self, tp_img, gt_img, label):
        if label == 0:
            return np.zeros((*tp_img.shape[:2], 3))
        elif label == 1 and gt_img is None:
            return np.full((*tp_img.shape[:2], 3), 255, dtype=np.uint8)
        else:
            return np.array(gt_img.convert('RGB'))

    def _process_masks(self, gt_img):
        gt_img = (np.mean(gt_img, axis=2, keepdims=True) > 127.5) * 1.0
        gt_img = gt_img.transpose(2, 0, 1)[0]
        masks_list = [gt_img]
        if self.edge_mask_generator:
            gt_img_edge = self.edge_mask_generator(gt_img)[0][0]
            masks_list.append(gt_img_edge)
        return masks_list

    def __getitem__(self, index):
        # random index
        idx = index
        if self.split == "train":
            # old_idx = idx # debug
            split_idx = idx // self.sampler_number
            idx_min = self.dataset_border[split_idx]
            idx_max = self.dataset_border[split_idx+1] - 1
            idx = random.randint(idx_min, idx_max) # 包含max
            # print(f"idx {old_idx}->{idx}") # debug
        
        index = idx
        sample = self.dataset[index]
        tp_path = sample['image']
        gt_path = sample['mask']
        tp_img = self.img_loader(tp_path)
        if tp_img is None: # 检查图片路径，mask不用检查，因为可能为"negative"
            raise ValueError(f"Image at {tp_path} could not be loaded.")
        if gt_path is not None:
            gt_img = self.img_loader(gt_path)
        else:
            gt_img = None
        tp_img = np.array(tp_img.convert('RGB'))
        gt_img = self._prepare_gt_img(tp_img, gt_img, sample['label'])

        if self.common_transforms:
            res_dict = self.common_transforms(image=tp_img, mask=gt_img)
            tp_img, gt_img = res_dict['image'], res_dict['mask']
            # import pdb;pdb.set_trace()
            if np.sum(gt_img>0):
                label=1
            else:
                label=0
            # label = sample['label']
        else:
            label = sample['label']
        # label = sample['label']

        # if label != sample['label']:
        #     print(sample['key'])
        #     print(sample['label'])
        #     print(label)

        masks_list = self._process_masks(gt_img)
        res_dict = self.post_transform(image=tp_img, masks=masks_list)

        data_dict = {
            'image_path': tp_path,
            'image': res_dict['image'],
            'mask': res_dict['masks'][0].unsqueeze(0),
            'label': label,
            'shape': torch.tensor(self.output_size if self.is_resizing else tp_img.shape[:2]),
            'name': sample['key']
        }

        if self.edge_mask_generator:
            data_dict['edge_mask'] = res_dict['masks'][1].unsqueeze(0)

        if self.is_padding:
            shape_mask = torch.zeros_like(data_dict['mask'])
            shape_mask[:, :data_dict['shape'][0], :data_dict['shape'][1]] = 1
            data_dict['shape_mask'] = shape_mask

        cat_net_post_func(data_dict)
        del sample
        del res_dict

        return data_dict

    def __len__(self):
        if self.split == "train":
            return self.sampler_number * self.num_datasets
        else:
            return len(self.dataset)