import glob
import os
import PIL
import h5py
import json
import torch
from torch.utils.data import Dataset


class H5VideoDataset(Dataset):
    """
    PyTorch Dataset for loading .h5 video files (T x H x W) from a given directory.

    Args:
        root_dir (str): Directory with all the .h5 files.
        case_ids (list of str, optional): List of sample ids to filter files. If provided, only files
                                          whose names contain one of the case_ids are used.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir, case_ids=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Get all .h5 files in the directory.
        all_files = glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True)

        # Filter files based on provided case_ids if any.
        if case_ids is not None:
            self.file_list = [
                f
                for f in all_files
                if any(case_id in os.path.basename(f) for case_id in case_ids)
            ]
        else:
            self.file_list = all_files

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # Load video data
        with h5py.File(file_path, "r") as f:
            video = f["images"][()]
        video = torch.from_numpy(video)  # Convert to tensor with shape (T x H x W)
        
        # Load corresponding info.json
        info_path = os.path.join(os.path.dirname(file_path), "info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
        else:
            info = None

        data_dict = {
            'video': video,
            'info': info,
            'file_path': file_path
        }

        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict


class H5SingleFrameDataset(Dataset):
    """
    PyTorch Dataset for loading individual frames from .h5 video files.
    Each item represents a single frame rather than a whole video.

    Args:
        root_dir (str): Directory with all the .h5 files.
        case_ids (list of str, optional): List of sample ids to filter files.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir, case_ids=None, transform=None, image_fmt='pth'):
        self.root_dir = root_dir
        self.image_fmt = image_fmt
        self.transform = transform
        # Get all .h5 files in the directory
        all_files = glob.glob(os.path.join(root_dir, "**", "*.h5"), recursive=True)
        
        # Filter files based on provided case_ids if any
        if case_ids is not None:
            self.file_list = [
                f for f in all_files 
                if any(case_id in os.path.basename(f) for case_id in case_ids)
            ]
        else:
            self.file_list = all_files
            
        # Create frame index mapping
        self.frame_idx_map = []  # List of (file_idx, frame_idx) tuples
        for file_idx, file_path in enumerate(self.file_list):
            with h5py.File(file_path, 'r') as f:
                n_frames = f['images'].shape[0]
                self.frame_idx_map.extend(
                    [(file_idx, frame_idx) for frame_idx in range(n_frames)]
                )

    def __len__(self):
        return len(self.frame_idx_map)

    def __getitem__(self, idx):
        file_idx, frame_idx = self.frame_idx_map[idx]
        file_path = self.file_list[file_idx]
        
        # Load single frame
        with h5py.File(file_path, 'r') as f:
            frame = f['images'][frame_idx]

        if self.image_fmt == 'pth':
            frame = torch.from_numpy(frame)
        elif self.image_fmt == 'npy':
            frame = frame 
        elif self.image_fmt == 'pil':
            frame = PIL.Image.fromarray(frame).convert('RGB')  # Convert to PIL Image
    
        # Load corresponding info.json
        info_path = os.path.join(os.path.dirname(file_path), "info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
        else:
            info = None
            
        data_dict = {
            'image': frame,
            'info': info,
            'file_path': file_path,
            'frame_idx': frame_idx
        }
        if self.transform:
            data_dict = self.transform(data_dict)

        return data_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Instantiate H5VideoDataset for OPTIMUM 28mm Sweeps"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/OPTIMUM/28mm_sweeps",
        help="Directory containing the .h5 files",
    )
    parser.add_argument(
        "--case_ids",
        type=str,
        nargs="*",
        default=None,
        help="List of case ids to filter the dataset, e.g. UA-006",
    )
    args = parser.parse_args()

    # Instantiate the dataset
    dataset = H5VideoDataset(args.root_dir, case_ids=args.case_ids)
    print(f"Loaded {len(dataset)} samples from {args.root_dir}")

    # Example: print info about video dataset
    if len(dataset) > 0:
        sample = dataset[0]
        print("\nVideo Dataset Sample:")
        print(f"Video shape: {sample['video'].shape}")  # Should be (T, H, W)
        print(f"Info: {sample['info']}")
        print(f"File path: {sample['file_path']}")
        
    # Try the single frame dataset
    frame_dataset = H5SingleFrameDataset(args.root_dir, case_ids=args.case_ids)
    print(f"\nLoaded {len(frame_dataset)} frames total")
    
    # Example: print info about single frame dataset
    if len(frame_dataset) > 0:
        sample = frame_dataset[0]
        print("\nSingle Frame Dataset Sample:")
        print(f"Frame shape: {sample['frame'].shape}")  # Should be (H, W)
        print(f"Frame index: {sample['frame_idx']}")
        print(f"File path: {sample['file_path']}")
