import os
from glob import glob

import numpy as np
from torch.utils.data import Dataset
from warnings import warn


warn("Deprecated: medAI.datasets.optimum.ua_unlabeled is deprecated", DeprecationWarning, stacklevel=2)


class UAUnlabeledImages(Dataset):
    """Gives access to the unlabeled images from the UA dataset.

    Items are dictionaries with the following keys:
    "bmode": the B-mode image: a 2D numpy array of shape (H, W)
        H - the height of the image (axial direction) with lower indices being further from the transducer (i.e. towards
            prostate anterior)
        W - the width of the image (lateral direction) with lower indices being further from the sonographer's hand
            (i.e. towards prostate base)
    "case_name": the case name

    Args:
        root (str, optional): the root directory of the dataset. If None, will be read from the environment variable
            UA_UNLABELED_IMAGES_ROOT. Defaults to None.
        transform (callable, optional): a function/transform that takes in an image and returns a transformed version.
        lazy_load (bool, optional): if True, the images will be loaded only when requested. Defaults to False.
    """

    def __init__(self, root=None, transform=None, lazy_load=False):
        if root is None:
            if "UA_UNLABELED_IMAGES_ROOT" not in os.environ:
                raise ValueError("The root directory of the dataset is not provided.")
            root = os.environ["UA_UNLABELED_IMAGES_ROOT"]
        self.root = root
        self.transform = transform

        self._images = {}
        self._case_names = []
        self._indices = []
        self._paths = []

        for i, path in enumerate(sorted(glob("*.npy", root_dir=root))):
            case_name = path.split("_")[0]
            self._case_names.append(case_name)
            self._paths.append(path)

            image = np.load(os.path.join(root, path), mmap_mode="r")
            if not lazy_load:
                self._images[i] = image 
            
            n_frames = image.shape[-1]
            self._indices.extend([(i, j) for j in range(n_frames)])

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        i, j = self._indices[idx]

        if not i in self._images:
            image = np.load(os.path.join(self.root, self._paths[i]), mmap_mode="r")
            self._images[i] = image
        
        image = self._images[i][..., j]
        image = np.flip(image, axis=0).copy()

        output = {"bmode": image, "case_name": self._case_names[i]}
        if self.transform:
            output = self.transform(output)
        return output
