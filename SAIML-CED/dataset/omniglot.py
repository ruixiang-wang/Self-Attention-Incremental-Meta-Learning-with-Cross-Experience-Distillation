import os
import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_url, check_integrity
from torch.utils.data import Dataset


class OmniglotDataset(Dataset):
    folder = 'omniglot-py'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    def __init__(self, root, background=True, transform=None, target_transform=None,
                 download=False, train=True, all=False):
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform
        self.images_cached = {}

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = os.path.join(self.root, self._get_target_folder())
        self._alphabets = os.listdir(self.target_folder)
        self._characters = sum([[os.path.join(a, c) for c in os.listdir(os.path.join(self.target_folder, a))]
                                for a in self._alphabets], [])
        self._character_images = [[(image, idx) for image in os.listdir(os.path.join(self.target_folder, character))
                                   if image.endswith('.png')]
                                  for idx, character in enumerate(self._characters)]
        self._flat_character_images = sum(self._character_images, [])
        self.data = [x[0] for x in self._flat_character_images]
        self.targets = [x[1] for x in self._flat_character_images]
        self.data2 = []
        self.targets2 = []
        self.new_flat = []
        for a in range(len(self.targets) // 20):
            start = a * 20
            if train:
                for b in range(start, start + 15):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
                    self.new_flat.append(self._flat_character_images[b])
            else:
                for b in range(start + 15, start + 20):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
                    self.new_flat.append(self._flat_character_images[b])

        if not all:
            self._flat_character_images = self.new_flat
            self.targets = self.targets2
            self.data = self.data2

        print("Total classes =", np.max(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data[index]
        character_class = self.targets[index]
        image_path = os.path.join(self.target_folder, self._characters[character_class], image_name)
        if image_path not in self.images_cached:
            image = Image.open(image_path, mode='r').convert('L')
            if self.transform:
                image = self.transform(image)
            self.images_cached[image_path] = image
        else:
            image = self.images_cached[image_path]

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class


