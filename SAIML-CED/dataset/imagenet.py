import os
import shutil
import tempfile
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import check_integrity, download_and_extract_archive


class CustomImageNet(ImageFolder):

    def __init__(self, root, split='train', download=False, **kwargs):
        root = self.root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", ("train", "val"))

        if download:
            self._download_data()
        wnid_to_classes = self._load_meta_file()[0]

        super(CustomImageNet, self).__init__(self.split_folder, **kwargs)
        self.root = root

        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}

    def _download_data(self):
        if not check_integrity(self.meta_file):
            tmp_dir = tempfile.mkdtemp()

            archive_dict = ARCHIVE_DICT['devkit']
            download_and_extract_archive(archive_dict['url'], self.root,
                                         extract_root=tmp_dir,
                                         md5=archive_dict['md5'])
            devkit_folder = _splitexts(os.path.basename(archive_dict['url']))[0]
            meta = self._parse_devkit(os.path.join(tmp_dir, devkit_folder))
            self._save_meta_file(*meta)

            shutil.rmtree(tmp_dir)

        if not os.path.isdir(self.split_folder):
            archive_dict = ARCHIVE_DICT[self.split]
            download_and_extract_archive(archive_dict['url'], self.root,
                                         extract_root=self.split_folder,
                                         md5=archive_dict['md5'])

            if self.split == 'train':
                self._prepare_train_folder(self.split_folder)
            elif self.split == 'val':
                val_wnids = self._load_meta_file()[1]
                self._prepare_val_folder(self.split_folder, val_wnids)
        else:
            msg = ("You set download=True, but a folder '{}' already exists in "
                   "the root directory. If you want to re-download or re-extract the "
                   "archive, delete the folder.")
            print(msg.format(self.split))

    @property
    def meta_file(self):
        return os.path.join(self.root, 'meta.bin')

    def _load_meta_file(self):
        if check_integrity(self.meta_file):
            return torch.load(self.meta_file)
        else:
            raise RuntimeError("Meta file not found or corrupted.",
                               "You can use download=True to create it.")

    def _save_meta_file(self, wnid_to_class, val_wnids):
        torch.save((wnid_to_class, val_wnids), self.meta_file)

    @property
    def split_folder(self):
        return os.path.join(self.root, self.split)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def _parse_devkit(self, root):
        idx_to_wnid, wnid_to_classes = self._parse_meta(root)
        val_indices = self._parse_val_groundtruth(root)
        val_wnids = [idx_to_wnid[idx] for idx in val_indices]
        return wnid_to_classes, val_wnids

    def parse_meta_info(devkit_root, path='data', filename='meta.mat'):
        metafile = os.path.join(devkit_root, path, filename)
        meta_data = sio.loadmat(metafile, squeeze_me=True)['synsets']
        nums_children = list(zip(*meta_data))[4]
        meta_data = [meta_data[idx] for idx, num_children in enumerate(nums_children)
                     if num_children == 0]
        indices, wnids, classes = list(zip(*meta_data))[:3]
        classes = [tuple(class_names.split(', ')) for class_names in classes]
        index_to_wnid = {idx: wnid for idx, wnid in zip(indices, wnids)}
        wnid_to_classes = {wnid: class_names for wnid, class_names in zip(wnids, classes)}
        return index_to_wnid, wnid_to_classes

    def parse_validation_groundtruth(devkit_root, path='data',
                                     filename='ILSVRC2012_validation_ground_truth.txt'):
        with open(os.path.join(devkit_root, path, filename), 'r') as txt_file:
            val_indices = txt_file.readlines()
        return [int(index) for index in val_indices]

    def prepare_train_data_folder(data_folder):
        for archive in [os.path.join(data_folder, archive) for archive in os.listdir(data_folder)]:
            extract_archive(archive, os.path.splitext(archive)[0], remove_finished=True)

    def prepare_validation_data_folder(data_folder, wnids):
        image_files = sorted([os.path.join(data_folder, file) for file in os.listdir(data_folder)])

        for wnid in set(wnids):
            os.mkdir(os.path.join(data_folder, wnid))

        for wnid, image_file in zip(wnids, image_files):
            shutil.move(image_file, os.path.join(data_folder, wnid, os.path.basename(image_file)))

    def split_extensions(root):
        extensions = []
        extension = '.'
        while extension:
            root, extension = os.path.splitext(root)
            extensions.append(extension)
        return root, ''.join(reversed(extensions))

