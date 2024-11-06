from heapq import merge
import torch.nn.functional as F
import torchvision.transforms as transforms 
from torch.utils.data import Dataset
from PIL import Image
import os


class AngiogramDataset(Dataset):
    def __init__(self, image_dir, max_masks=1, load_to_device=None):
        self.image_dir = os.path.join(image_dir , "images")
        self.mask_dir = os.path.join(image_dir , "masks")
        self.image_paths = os.listdir(self.image_dir)

        self.max_masks = max_masks
        self.tensor = transforms.ToTensor()
        self.load_to_device = load_to_device

    def __len__(self):
        return len(self.image_paths)
    def get_image_name(self, idx):
        return self.image_paths[idx]

    def __getitem__(self, index):
        img = self.image_paths[index]
        image_path = os.path.join(self.image_dir , img) 
        mask_path = os.path.join(self.mask_dir, img)

        image = Image.open(image_path).convert('L')
        masks = Image.open(mask_path).convert('L') 
        # cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # image = torch.tensor(np.array(image)).float()
        image = self.tensor(image)
        masks = self.tensor(masks)

        if image.ndim == 2:  # Adds a channel dimension if grayscale
            image = image.unsqueeze(0)
        if masks.ndim == 2:  # Adds a channel dimension if grayscale
            masks = masks.unsqueeze(0)

        # handel legacy multi-mask model needs.
        if self.max_masks != 1 and masks.shape[0] < self.max_masks:
            padding = self.max_masks - masks.shape[0]
            pad_width = (0, 0, 0, 0, 0, padding)
            masks = F.pad(masks, pad_width, value=0)
        else:
            masks = masks[:self.max_masks,:,:]

        if self.load_to_device:
            image = image.to(self.load_to_device)
            masks = masks.to(self.load_to_device)

        return image, masks

class ImageLoader:
    def __init__(self, load_to_device=None) -> None:
        self.tensor = transforms.ToTensor()
        self.pil = transforms.ToPILImage()
        self.load_to_device = load_to_device

    def load(self, image_path:str):
        image = Image.open(image_path).convert('L')
        image.resize((512, 512), Image.Resampling.BICUBIC)
        image = self.tensor(image)

        if image.ndim == 2:  # Adds a channel dimension if grayscale
            image = image.unsqueeze(0)

        if self.load_to_device:
            image = image.to(self.load_to_device)

        return image
    def convert_to_pil(self, image) -> Image.Image:
        merged = 0
        for m in image:
            merged += m
        return self.pil(merged)

