"""
Dataset loading for FER2013, expected on disk as:

    data_dir/
        train/
            angry/    *.jpg
            disgust/
            fear/
            happy/
            sad/
            surprise/
            neutral/
        test/
            angry/
            ...

This is the standard layout the Kaggle FER2013 (in-image form) ships in.
"""

import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

IMG_HEIGHT, IMG_WIDTH = 48, 48
EXPRESSIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class FERDataset(Dataset):
    """Reads (image_path, label) pairs from an FER2013-style folder tree."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for label_idx, emotion in enumerate(EXPRESSIONS):
            emotion_dir = os.path.join(root_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            for fname in sorted(os.listdir(emotion_dir)):
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(emotion_dir, fname), label_idx))

        if not self.samples:
            raise RuntimeError(f"No images found under {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label


class _WithTransform(Dataset):
    """Applies a transform to a Subset whose base dataset has transform=None.

    Needed because train/val come from the same folder split but need
    different augmentation, and random_split() only gives you index subsets
    of a single underlying dataset.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def build_transforms():
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return train_transform, eval_transform


def create_data_loaders(data_dir, batch_size, val_split=0.2, num_workers=0, seed=123):
    train_root = os.path.join(data_dir, "train")
    test_root = os.path.join(data_dir, "test")
    if not os.path.isdir(train_root) or not os.path.isdir(test_root):
        raise FileNotFoundError(f"Expected 'train/' and 'test/' inside {data_dir}")

    train_transform, eval_transform = build_transforms()

    base_train = FERDataset(train_root, transform=None)
    n_val = int(len(base_train) * val_split)
    n_train = len(base_train) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_train, [n_train, n_val], generator=generator)

    train_ds = _WithTransform(train_subset, train_transform)
    val_ds = _WithTransform(val_subset, eval_transform)
    test_ds = FERDataset(test_root, transform=eval_transform)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(num_workers=num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(
        f"train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)} "
        f"| classes: {EXPRESSIONS}"
    )
    return train_loader, val_loader, test_loader
