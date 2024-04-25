import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class TeethSegDataset(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.cur_path = config.CUR_PATH
        self.subset = config.subset
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        
        self.sample_points_num = config.npoints

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'TeethSeg3D')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'TeethSeg3D')
    
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()

        self.file_list = []
        for line in lines:
            line = line.strip()
            self.file_list.append(line)
        
        self.file_list = self.file_list[:int(len(self.file_list))]
        
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'TeethSeg3D')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
    
    def __getitem__(self, idx):
        points = IO.get(os.path.join(self.pc_path, self.file_list[idx])).astype(np.float32)
        curvatures = IO.get(os.path.join(self.cur_path, self.file_list[idx][:-4]+'.npy')).astype(np.float32)
        
        points_norm = self.pc_norm(points)
        selected_idxs = np.random.choice(len(points_norm), self.sample_points_num, replace=True)
        sampled_points = points_norm[selected_idxs]
        sampled_curs = curvatures[selected_idxs]

        sampled_points = torch.from_numpy(sampled_points).float()
        sampled_curs = torch.from_numpy(sampled_curs).float()

        return sampled_points, sampled_curs

    def __len__(self):
        return len(self.file_list)


@DATASETS.register_module()
class TeethSegFinetuneDataset(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.gt_path = config.GT_PATH
        self.subset = config.subset
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}_finetune.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'TeethSeg3D')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'TeethSeg3D')

        self.label2id = {0:0, \
                        11:1, 12:2, 13:3, 14:4, 15:5, 16:6, 17:7, 18:8, \
                        21:9, 22:10, 23:11, 24:12, 25:13, 26:14, 27:15, 28:16, \
                        31:1, 32:2, 33:3, 34:4, 35:5, 36:6, 37:7, 38:8, \
                        41:9, 42:10, 43:11, 44:12, 45:13, 46:14, 47:15, 48:16}
    
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        self.file_list = []

        for line in lines:
            line = line.strip()
            mesh_id = line.split('_')[0]
            location = line.split('_')[1].split('.')[0]
            location = 0 if location == 'lower' else 1
            self.file_list.append({
                'location': location,
                'mesh_id': mesh_id,
                'file_path': line
            })

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'TeethSeg3D')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc, centroid, m
  
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        # load points and labels
        points = IO.get(os.path.join(self.pc_path, sample['mesh_id'], sample['file_path'])).astype(np.float32)
        cls = sample['location']
        labels = IO.get(os.path.join(self.gt_path, sample['mesh_id'], sample['file_path'].replace('obj', 'json')))['labels']
        labels = np.array([self.label2id[label] for label in labels]).astype(np.int32)

        # normalization
        points_norm, center, scale = self.pc_norm(points)

        # random sample
        selected_idxs = np.random.choice(len(points_norm), self.sample_points_num, replace=True)
        sampled_points = points_norm[selected_idxs]
        sampled_labels = labels[selected_idxs]

        sampled_points = torch.from_numpy(sampled_points).float()
        sampled_labels = torch.from_numpy(sampled_labels).long()

        # get the class weight
        class_weights = torch.zeros((17)).float()
        tmp, _ = torch.histogram(sampled_labels.float(), bins=17, range=(0., 17.))
        class_weights += tmp
        class_weights = class_weights / torch.sum(class_weights)
        class_weights = torch.where(torch.isinf(class_weights), torch.full_like(class_weights, 0), class_weights)

        if self.subset == 'test':
            points = torch.from_numpy(points).float()
            labels = torch.from_numpy(labels).long()
            center, scale = torch.tensor(center).float(), torch.tensor(scale).float()
            return sampled_points, cls, sampled_labels, points, labels, center, scale, class_weights
        else:
            return sampled_points, cls, sampled_labels, class_weights

    def __len__(self):
        return len(self.file_list)