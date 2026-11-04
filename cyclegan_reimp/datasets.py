

import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(folder):
    return sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTENSIONS)
    )


class UnalignedDataset(Dataset):
    

    def __init__(self, dataroot, phase="train", load_size=286, crop_size=256, is_train=True):
        self.dir_A = os.path.join(dataroot, phase + "A")
        self.dir_B = os.path.join(dataroot, phase + "B")
        self.paths_A = list_images(self.dir_A)
        self.paths_B = list_images(self.dir_B)
        self.size_A = len(self.paths_A)
        self.size_B = len(self.paths_B)
        assert self.size_A > 0 and self.size_B > 0, (
            f"No images found in {self.dir_A} or {self.dir_B}. "
            f"Populate {phase}A/ and {phase}B/ with domain images."
        )

        
        if is_train:
            self.transform = T.Compose([
                T.Resize(load_size, T.InterpolationMode.BICUBIC),
                T.RandomCrop(crop_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # -> [-1, 1]
            ])
        else:
            self.transform = T.Compose([
                T.Resize((crop_size, crop_size), T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return max(self.size_A, self.size_B)

    def __getitem__(self, idx):
        path_A = self.paths_A[idx % self.size_A]
        idx_B = random.randint(0, self.size_B - 1)  # unpaired random sampling
        path_B = self.paths_B[idx_B]

        img_A = Image.open(path_A).convert("RGB")
        img_B = Image.open(path_B).convert("RGB")

        return {
            "A": self.transform(img_A),
            "B": self.transform(img_B),
            "path_A": path_A,
            "path_B": path_B,
        }
