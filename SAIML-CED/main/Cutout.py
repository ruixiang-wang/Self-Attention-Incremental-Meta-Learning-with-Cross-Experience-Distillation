import numpy as np
import torch

class Cutout(object):
    def __init__(self, num_holes, patch_length):
        self.num_holes = num_holes
        self.patch_length = patch_length

    def __call__(self, image):
        height = 32
        width = 32

        mask = np.ones((height, width), np.float32)

        for n in range(self.num_holes):
            y = np.random.randint(height)
            x = np.random.randint(width)

            y1 = np.clip(y - self.patch_length // 2, 0, height)
            y2 = np.clip(y + self.patch_length // 2, 0, height)
            x1 = np.clip(x - self.patch_length // 2, 0, width)
            x2 = np.clip(x + self.patch_length // 2, 0, width)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(image)
        image = image * mask

        return image
