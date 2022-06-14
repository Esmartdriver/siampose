# Motion JPEG dataloader. Allows reading motion JPEG videos as unlabelled data.
from re import I
from pytest import param
import torch
import glob
import os
from natsort import natsorted
import cv2
import os
from tqdm import tqdm
import random
import numpy as np
import pytorch_lightning
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import json

IMAGENET_MEAN_STD = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class MjpegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        transform=None,
        single_image_mode=False,
        crop_height=64,
        crop_width=64,
        min_scale=1,
        max_scale=5,
        center_sampling="uniform",
        only_crops=True,
        include_labelled=False,
        labelled_only=False,
    ) -> None:
        self.center_sampling = center_sampling

        if labelled_only:
            self.unlabelled_files = []
            include_labelled = True
        else:
            self.unlabelled_files = natsorted(glob.glob(os.path.join(path, "unlabelled/*.jpg")))
        self.files = []
        self.labelled_files_map = {}
        if include_labelled:
            for folder in natsorted(glob.glob(os.path.join(path, "*"))):
                if os.path.isdir(folder):
                    category = folder.split("/")[-1]
                    if category == "unlabelled":
                        continue
                    self.labelled_files_map[category] = natsorted(glob.glob(os.path.join(path, f"{category}/*.jpg")))

        self.single_image_mode = single_image_mode
        self.file_map = {}
        self.crop_width, self.crop_height = crop_width, crop_height
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.only_crops = only_crops
        # self.labelled_only = labelled_only
        self._build_index()
        self._build_sequence_id_map()
        self._build_file_map()

        self.transform = transform

    def _build_index(self):
        self.files = self.unlabelled_files
        for category in self.labelled_files_map:

            self.files += self.labelled_files_map[category]

    def random_bbox_crop(self, img, x1, y1, x2, y2):
        if not isinstance(img, np.ndarray):
            im_h, im_w = img.height, img.width
        else:
            im_h, im_w, _ = img.shape
        xc, yc = (x1 + x2) // 2, (y1 + y2) // 2
        h, w = abs(x2 - x1), abs(y2 - y1)
        # CROP_WITDH, CROP_HEIGHT = 32, 32
        MIN_SCALE = 1
        MAX_SCALE = 4
        crop_w = int(np.clip(np.random.uniform(w * MIN_SCALE, w * MAX_SCALE), 0, im_h))
        crop_h = int(np.clip(np.random.uniform(h * MIN_SCALE, h * MAX_SCALE), 0, im_w))
        x1, y1 = int(np.clip(xc - crop_w / 2, 0, im_w - 1)), int(np.clip(yc - crop_h / 2, 0, im_h - 1))
        x2, y2 = int(np.clip(xc + crop_w / 2, 0, im_w - 1)), int(np.clip(yc + crop_h / 2, 0, im_h - 1))
        if isinstance(img, np.ndarray):
            crop = img[y1:y2, x1:x2]
        else:  # Assume Pillow Image
            img: Image.Image
            crop = img.crop([x1, y1, x2, y2])
        return crop

    def random_video_crop(self, img, xc, yc):
        if not isinstance(img, np.ndarray):
            h, w = img.height, img.width
        else:
            h, w, _ = img.shape
        crop_w = int(np.clip(np.random.uniform(self.crop_width * self.min_scale, self.crop_width * self.max_scale), 0, h))
        crop_h = int(np.clip(np.random.uniform(self.crop_height * self.min_scale, self.crop_height * self.max_scale), 0, w))
        x1, y1 = int(np.clip(xc - crop_w / 2, 0, w - 1)), int(np.clip(yc - crop_h / 2, 0, h - 1))
        x2, y2 = int(np.clip(xc + crop_w / 2, 0, w - 1)), int(np.clip(yc + crop_h / 2, 0, h - 1))
        if isinstance(img, np.ndarray):
            crop = img[y1:y2, x1:x2]
        else:  # Assume Pillow Image
            img: Image.Image
            crop = img.crop([x1, y1, x2, y2])
        return crop

    def __len__(self):
        return len(self.files)

    def _build_sequence_id_map(self):
        self.sequence_id_map = {}
        for file in self.files:
            sequence_id = self._get_sequence_id(file)
            if sequence_id not in self.sequence_id_map:
                self.sequence_id_map[sequence_id] = [file]
            else:
                self.sequence_id_map[sequence_id].append(file)

    def _build_file_map(self):
        self.file_map = {}
        self.sequence_uid_map = {}
        # sequence_uid=0
        for idx, file in enumerate(tqdm(self.unlabelled_files)):
            basename = os.path.basename(file)
            sequence_id = self._get_sequence_id(file)
            self.file_map[basename] = {
                "file": file,
                "sequence_id": sequence_id,
                "frame_number": self._get_frame_number(file),
                "frame_index": self.sequence_id_map[sequence_id].index(file),
                "idx": idx,
            }
            # if sequence_id not in

    def _get_sequence_id(self, filename):
        basename = os.path.basename(filename)
        category = self._get_category(filename)
        if category == "unlabelled":
            sequence_id = "_".join(basename.split("_")[0:-1])
        else:  # Remove the "object ID" also
            sequence_id = "_".join(basename.split("_")[0:-2])
        return sequence_id

    def _get_frame_number(self, filename):
        basename = os.path.basename(filename)
        category = self._get_category(filename)
        if category == "unlabelled":
            frame_number = basename.split("_")[-1].replace(".jpg", "")
        else:  # Remove the "object ID" also
            frame_number = basename.split("_")[-2].replace(".jpg", "")

        return int(float(frame_number))

    def get_single_item(self, idx):
        filename = self.files[idx]
        return self.get_single_file(filename)

    def get_single_file(self, filename):
        # img = self._fast_read(filename)
        img = Image.open(filename)
        category = self._get_category(filename)
        sample = {
            "image": img,
            "sequence": self._get_sequence_id(filename),
            "path": filename,
            "frame_numer": self._get_frame_number(filename),
            "category": category,
            "category_id": self.categories.index(category),
            "filename": filename,
        }
        return sample

    def _get_category(self, filename):
        return filename.split("/")[-2]

    def get_frame_index(self, filename):
        basename = os.path.basename(filename)
        frame_index = self.file_map[basename]["frame_index"]
        return frame_index

    def _get_nearby_frame_index(self, filename, min_dist=1, max_dist=5, both_ways=True):
        basename = os.path.basename(filename)
        frame_index = self.file_map[basename]["frame_index"]
        gap = random.randint(min_dist, max_dist)
        sequence_id = self._get_sequence_id(filename)
        if both_ways and random.random() > 0.5:
            gap = -gap
        assert gap != 0
        last_index = len(self.sequence_id_map[sequence_id]) - 1
        nearby_frame_index = np.clip(frame_index + gap, 1, last_index)
        if nearby_frame_index == frame_index:
            nearby_frame_index = np.clip(frame_index - gap, 1, last_index)
        return nearby_frame_index

    @property
    def seq_subset(self):
        return natsorted(list(self.sequence_id_map))

    @property
    def categories(self):
        categories_list = ["unlabelled"]
        categories_list += natsorted(list(self.labelled_files_map))
        return categories_list

    def random_uniform_center(self, w, h):
        xc = int(np.random.uniform(0, w))
        yc = int(np.random.uniform(0, h))
        return xc, yc

    def random_normal_center(self, w, h, std_dev=3):
        xc = int(np.clip(np.random.normal(w / 2, w / std_dev), 0, w - 1))
        yc = int(np.clip(np.random.normal(h / 2, h / std_dev), 0, h - 1))
        return xc, yc

    def __getitem__(self, idx):
        base_sample = self.get_single_item(idx)
        if self.single_image_mode:
            if self.transform:
                base_sample["image"] = self.transform(base_sample["image"])
            return base_sample
        else:
            # Crop pairs!
            sample = base_sample

            img1 = sample["image"]
            json_file = sample["filename"].replace(".jpg", ".json")
            prediction = None
            if os.path.exists(json_file) and random.random() < 0.5:
                with open(json_file) as f:
                    predictions = json.load(f)
                if len(predictions) > 0:
                    prediction = random.sample(predictions, 1)[0]
            if prediction is None:
                crop1, crop2 = self.random_video_pair_crops(sample, img1)
            else:
                x1, y1, x2, y2 = prediction["bbox"]
                crop1 = self.random_bbox_crop(img1, x1, y1, x2, y2)

                nearby_index = self._get_nearby_frame_index(sample["path"])
                nearby_filename = self.sequence_id_map[sample["sequence"]][nearby_index]
                # img2 = Image.open(nearby_filename)
                crop2 = self.random_bbox_crop(img1, x1, y1, x2, y2)
            crops = [crop1, crop2]
            if self.transform:
                crops = [self.transform(crop) for crop in crops]
            assert len(crops) == 2
            sample["crops"] = crops
            if self.only_crops:
                del sample["image"]
            return sample

    def random_video_pair_crops(self, sample, img1):
        if isinstance(img1, np.ndarray):
            h, w, c = img1.shape
        else:
            img1: Image.Image
            h, w = img1.height, img1.width
        if self.center_sampling == "uniform":
            xc, yc = self.random_uniform_center(w, h)
        elif self.center_sampling == "normal":
            xc, yc = self.random_normal_center(w, h)
        else:
            raise NotADirectoryError(f"Sampling {self.center_sampling} not enabled.")
        crop1 = self.random_video_crop(img1, xc, yc)
        nearby_index = self._get_nearby_frame_index(sample["path"])
        nearby_filename = self.sequence_id_map[sample["sequence"]][nearby_index]
        img2 = Image.open(nearby_filename)
        crop2 = self.random_video_crop(img2, xc, yc)
        return crop1, crop2


class MjpegDataModule(pytorch_lightning.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=512,
        image_size=224,
        num_workers=8,
        pairing="next",
        dryrun=False,
        generate_embeddings=False,  # Usefull when generating embeddings
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.issetup = False
        self.num_workers = num_workers
        self.pairing = pairing
        self.dryrun = dryrun
        self.generate_embeddings = generate_embeddings

    def setup(self, stage=None):
        if not self.issetup:
            self.train_transform = self.get_transform(self.image_size)
            self.eval_transform = self.get_transform(self.image_size, evaluation=True)
            if not self.generate_embeddings:
                self.train_dataset = MjpegDataset(
                    os.path.join(self.data_dir, "training"),
                    transform=self.train_transform,
                )
                self.val_dataset = MjpegDataset(
                    os.path.join(self.data_dir, "validation"),
                    transform=self.eval_transform,
                )
                self.test_dataset = MjpegDataset(
                    os.path.join(self.data_dir, "test"),
                    transform=self.eval_transform,
                )
            else:
                self.train_dataset = MjpegDataset(
                    os.path.join(self.data_dir, "training"),
                    transform=self.train_transform,
                    labelled_only=True,
                    single_image_mode=True,
                )
                self.val_dataset = MjpegDataset(
                    os.path.join(self.data_dir, "validation"),
                    transform=self.eval_transform,
                    labelled_only=True,
                    single_image_mode=True,
                )
                self.test_dataset = MjpegDataset(
                    os.path.join(self.data_dir, "test"),
                    transform=self.eval_transform,
                    labelled_only=True,
                    single_image_mode=True,
                )
            self.train_sample_count = len(self.train_dataset)
            self.valid_sample_count = len(self.val_dataset)
            self.issetup = True

    def get_transform(self, image_size, mean_std=IMAGENET_MEAN_STD, evaluation=False):
        if not evaluation:
            # hflip = T.RandomHorizontalFlip()
            p_blur = 0.5 if image_size > 32 else 0
            transform_list = [
                # T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                # hflip,
                #
                T.Resize((image_size, image_size)),
                T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=image_size // 20 * 2 + 1, sigma=(0.1, 2.0))],
                    p=p_blur,
                ),
                T.ToTensor(),
                T.Normalize(*mean_std),
            ]
            return T.Compose(transform_list)
        else:
            return T.Compose(
                [
                    T.Resize((image_size, image_size)),
                    T.ToTensor(),
                    T.Normalize(*mean_std),
                ]
            )

    def train_dataloader(self, evaluation=False) -> DataLoader:
        self.train_dataset.transform = self.train_transform
        if evaluation:
            self.train_dataset.transform = self.eval_transform
            self.train_dataset.memory = True  # Avoid horizontal flip for evaluation.
        if self.dryrun:  # Just to quickly test the training loop. Trains on "test set", validation on valid set.
            print("WARNING: DRY RUN. Not performing real training.")
            return self.test_dataloader()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self, evaluation=False) -> DataLoader:
        self.val_dataset.memory = False
        if evaluation:
            self.val_dataset.memory = True  # Avoid horizontal flip for evaluation.
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
