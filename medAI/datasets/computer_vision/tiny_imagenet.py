from torch.utils.data import Dataset
from os.path import join 
from PIL import Image
import pandas as pd
import glob 


class TinyImageNetDataset(Dataset):
    def __init__(self, path: str, split='train', transform=None):
        self.path = path 
        self.transform = transform
        self.split = split

        wnid2desc = {}
        with open(join(path, 'words.txt'), 'r') as f:
            for line in f:
                wnid, desc = line.strip().split('\t')
                wnid2desc[wnid] = desc
        self.wnid2desc = wnid2desc
       
        wnids = []
        with open(join(path, 'wnids.txt'), 'r') as f:
            for line in f:
                wnid = line.strip()
                wnids.append(wnid)

        self.wnids = sorted(wnids)
        self.wnid2idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        if self.split == 'train': 
            labels = []
            image_paths = [] 

            for wnid in self.wnids: 
                for img_path in sorted(glob.glob(join(path, 'train', wnid, 'images', '*.JPEG'))): 
                    labels.append(self.wnid2idx[wnid])
                    image_paths.append(img_path)

            self.labels = labels
            self.image_paths = image_paths
            
        elif self.split == 'val': 
            labels = [] 
            image_paths = [] 
            with open(join(path, 'val', 'val_annotations.txt'), 'r') as f:
                for line in f: 
                    img_fname, wnid, *_ = line.strip().split('\t')
                    labels.append(self.wnid2idx[wnid])
                    image_paths.append(join(path, 'val', 'images', img_fname))

            self.labels = labels
            self.image_paths = image_paths
        
        else: 
            raise ValueError(f"Invalid split: {self.split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

