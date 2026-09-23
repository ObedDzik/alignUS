import json
import os
from PIL import Image 
import pandas as pd
from torchvision.datasets import ImageFolder


class BModeImageFolder:
    def __init__(self, root, split='train', fold=0, transform=None, splits_file=None): 
        self.root = root 
        self.transform = transform
        with open(os.path.join(root, 'dataset_info.json'), 'r') as f:
            dataset_info = json.load(f)
        self.id_column = dataset_info['id_column']
        self.array_data_sources = dataset_info['array_data_sources']

        self.metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))

        splits_file = splits_file or os.path.join(root, 'splits.json')
        with open(splits_file, 'r') as f:
            splits = json.load(f)
            self.ids = splits[fold][split]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        out = {}
        id_ = self.ids[idx]
        out['id'] = id_

        metadata = self.metadata[self.metadata[self.id_column] == id_].iloc[0].to_dict()
        out.update(metadata)

        for source in self.array_data_sources:
            image = Image.open(os.path.join(self.root, source, f"{id_}.png"))
            out[source] = image

        if self.transform:
            out = self.transform(out)

        return out


class ImageFolderSplits(ImageFolder): 
    def __init__(self, root, splits_file, split='train', **kwargs):

        with open(os.path.join(root, splits_file)) as f: 
            ids = json.load(f)[split]
        
        super().__init__(root, **kwargs)
        new_samples = []

        samples_dict = {
            os.path.basename(sample[0]): sample for sample in self.samples
        }
        for id in ids: 
            for key in list(samples_dict.keys()): 
                if key.startswith(id): 
                    new_samples.append(samples_dict.pop(key))
            
        self.samples = new_samples


