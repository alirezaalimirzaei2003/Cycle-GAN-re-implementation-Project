

import random
import numpy as np
import torch
from PIL import Image


class ImagePool:
    

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.num_imgs = 0
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.detach(), 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return torch.cat(return_images, 0)


def linear_lr_lambda(epoch, n_epochs=100, n_epochs_decay=100, epoch_count=1):
    
    return 1.0 - max(0, epoch + epoch_count - n_epochs) / float(n_epochs_decay + 1)


def tensor2image(tensor):
    
    img = tensor.detach().cpu().float().numpy()
    img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
    return img.clip(0, 255).astype(np.uint8)


def save_image(tensor, path):
    Image.fromarray(tensor2image(tensor)).save(path)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for p in net.parameters():
                p.requires_grad = requires_grad
