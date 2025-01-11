import os
import torch
from torch.utils.data import DataLoader
from custom_celeb_a import CustomCelebA
from utils.transforms import get_transforms

def custom_collate(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def get_dataloaders(batch_size=64):
    root_dir = 'D:/1111/studia/2sem/DeepLearning/zad1/data/celeba'
    img_dir = os.path.join(root_dir, 'img_align_celeba')
    attr_file = os.path.join(root_dir, 'list_attr_celeba.txt')
    eval_file = os.path.join(root_dir, 'list_eval_partition.txt')
    
    assert os.path.exists(root_dir), f"Root directory {root_dir} does not exist."
    assert os.path.exists(img_dir), f"img_align_celeba directory does not exist."
    assert os.path.exists(attr_file), f"list_attr_celeba.txt does not exist."
    assert os.path.exists(eval_file), f"list_eval_partition.txt does not exist."

    print(f"Loading CustomCelebA dataset from {root_dir}...")
    
    train_dataset = CustomCelebA(root_dir=root_dir, split="train", transform=get_transforms())
    val_dataset = CustomCelebA(root_dir=root_dir, split="val", transform=get_transforms())

    # Create DataLoaders with custom_collate
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    
    return train_loader, val_loader
