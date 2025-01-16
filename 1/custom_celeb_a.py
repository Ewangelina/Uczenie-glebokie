import os
from torch.utils.data import Dataset
from PIL import Image

class CustomCelebA(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Paths to the required files
        self.image_dir = os.path.join(root_dir, "img_align_celeba")
        self.attr_file = os.path.join(root_dir, "list_attr_celeba.txt")
        self.eval_file = os.path.join(root_dir, "list_eval_partition.txt")

        # Load attributes and partition information
        with open(self.attr_file, "r") as f:
            lines = f.readlines()
        self.attributes = [line.strip().split() for line in lines[2:]]  # Skip header

        with open(self.eval_file, "r") as f:
            eval_lines = f.readlines()
        self.partitions = [line.strip().split()[1] for line in eval_lines]

        # Map split to partition index
        split_mapping = {"train": 0, "val": 1, "test": 2}
        if split not in split_mapping:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        partition_idx = split_mapping[split]

        # Filter images and attributes by the split
        self.image_paths = [
            os.path.join(self.image_dir, attr[0])
            for attr, partition in zip(self.attributes, self.partitions)
            if int(partition) == partition_idx
        ]
        self.labels = [
            int(attr[21])  # Use attribute "Male" (index 20, starting from 0)
            for attr, partition in zip(self.attributes, self.partitions)
            if int(partition) == partition_idx
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
    
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
    
        # Retrieve label
        label = 1 if self.labels[idx] == 1 else 0  # Convert label to binary
    
        return image, label, img_path  # Include the image path
