from calendar import c
from multiprocessing.context import assert_spawning

from pytest import skip
from siampose.data.classification.mjpeg import MjpegDataset, MjpegDataModule
import numpy as np
import torch
import os
import cv2
from PIL import Image

INTERACTIVE = False


def test_test():
    assert True


def test_random_video_crop():
    dataset = MjpegDataset("test/data/mjpeg/training", single_image_mode=True)
    img = dataset[0]["image"]
    if not isinstance(img, np.ndarray):
        h, w = img.height, img.width
    else:
        h, w, c = img.shape
    crop = dataset.random_video_crop(img, w // 2, h // 2)
    if not isinstance(img, np.ndarray):
        crop_w, crop_h = crop.width, crop.height
    else:
        crop_h, crop_w, _ = crop.shape

    assert dataset.min_scale * crop_w < crop_w < dataset.max_scale * crop_w
    assert dataset.min_scale * crop_h < crop_h < dataset.max_scale * crop_h


def test_dataset_get_sequence_id_and_frame_number():
    dataset = MjpegDataset("test/data/mjpeg/training", single_image_mode=True)
    sequence_id = dataset._get_sequence_id("unlabelled/long_sequence_name_00005.jpg")
    assert sequence_id == "long_sequence_name"
    frame_number = dataset._get_frame_number("bogus_category/some_string_with_underscores_12345_6.jpg")
    assert frame_number == 12345


def test_get_nearby_frame_index():
    dataset = MjpegDataset("test/data/mjpeg/training", single_image_mode=True)
    filename = "test/data/mjpeg/training/unlabelled/20210723_172608_031446.jpg"
    idx = dataset.get_frame_index(filename)
    seq_id = dataset._get_sequence_id(filename)
    nearby_idx = dataset._get_nearby_frame_index(
        "test/data/mjpeg/training/unlabelled/20210723_172608_031446.jpg", min_dist=1, max_dist=1
    )
    assert idx != nearby_idx
    assert abs(idx - nearby_idx) <= 1
    assert dataset.sequence_id_map[seq_id][dataset.get_frame_index(filename)] == filename


def test_dataset_unlabelled_single_mode():
    dataset = MjpegDataset("test/data/mjpeg/training", single_image_mode=True, only_crops=False)
    assert len(dataset.unlabelled_files) > 0
    assert len(dataset) == 11
    assert "image" in dataset[0]
    assert "sequence" in dataset[0]
    assert dataset[0]["sequence"] == "20210723_172608"
    assert os.path.basename(dataset[0]["path"]) != dataset[0]["path"]
    assert len(dataset.sequence_id_map) == 4
    assert len(dataset.file_map) == len(dataset)

    for sample in dataset:
        assert "image" in sample
        assert isinstance(sample["image"], Image.Image)
        if INTERACTIVE:
            cv2.imshow("test", sample["image"])
            cv2.waitKey(1)


import torchvision.transforms as transforms


def test_dataset_unlabelled():
    dataset = MjpegDataset("test/data/mjpeg/training", transform=None)
    assert len(dataset.unlabelled_files) > 0
    assert len(dataset) == 11
    sample = dataset[0]
    assert "sequence" in sample
    assert "crops" in sample
    crop1, crop2 = sample["crops"]
    assert crop1.width != crop2.width
    assert isinstance(crop1, Image.Image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ]
    )
    dataset = MjpegDataset("test/data/mjpeg/training", transform=transform)
    assert isinstance(dataset.seq_subset, list)
    assert len(dataset.seq_subset) > 0
    assert isinstance(dataset.seq_subset[0], str)
    sample = dataset[0]
    crop1, crop2 = sample["crops"]
    assert sample["category"] == "unlabelled"
    assert isinstance(crop1, torch.Tensor)
    assert crop1.shape == crop2.shape
    assert crop1.shape == (3, 224, 224)


def test_dataset_labelled():
    dataset = MjpegDataset("test/data/mjpeg/training", transform=None, include_labelled=True, single_image_mode=True)
    assert len(dataset.unlabelled_files) > 0
    assert len(dataset) == 17
    sample = dataset[0]
    assert "sequence" in sample
    assert "crops" not in sample
    assert "image" in sample
    image = sample["image"]
    assert isinstance(image, Image.Image)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ]
    )
    dataset = MjpegDataset("test/data/mjpeg/training", transform=transform, include_labelled=True, single_image_mode=True)
    assert isinstance(dataset.seq_subset, list)
    assert len(dataset.seq_subset) > 0
    assert isinstance(dataset.seq_subset[0], str)
    sample = dataset[-1]
    image = sample["image"]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert sample["category"] == "speed_limit"


def test_dataset_labelled_only():
    dataset = MjpegDataset(
        "test/data/mjpeg/training", transform=None, include_labelled=True, labelled_only=True, single_image_mode=True
    )
    assert len(dataset.unlabelled_files) > 0
    assert len(dataset) == 6
    sample = dataset[0]
    assert "sequence" in sample
    assert "crops" not in sample
    assert "image" in sample
    image = sample["image"]
    assert isinstance(image, Image.Image)
    for sample in dataset:
        assert sample["category"] != "unlabelled"
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ]
    )
    dataset = MjpegDataset(
        "test/data/mjpeg/training", transform=transform, include_labelled=True, labelled_only=True, single_image_mode=True
    )
    assert isinstance(dataset.seq_subset, list)
    assert len(dataset.seq_subset) > 0
    assert isinstance(dataset.seq_subset[0], str)
    sample = dataset[-1]
    image = sample["image"]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
    assert sample["category"] == "speed_limit"
    for sample in dataset:
        assert sample["category"] != "unlabelled"


def test_mjpeg_data_module():
    dm = MjpegDataModule("test/data/mjpeg")
    dm.setup()
    assert dm is not None
    sample = dm.train_dataset[0]
    assert sample is not None
    for sample in dm.train_dataset:
        assert sample["crops"] is not None
    dataloader = dm.train_dataloader()
    for sample in dataloader:
        assert sample["crops"] is not None
        crop1, crop2 = sample["crops"]
        assert crop1.shape == (11, 3, 224, 224)
        assert crop2.shape == (11, 3, 224, 224)


import pytest


@pytest.mark.skip(reason="Too long to test.")
def test_mjpeg_data_module_full():
    dm = MjpegDataModule("/home/raphael/esmart/esmart-ai-datasets/data/esmart_mjpeg/")
    dm.setup()
    assert dm is not None
    for sample in dm.train_dataset:
        assert sample["crops"] is not None
        crop1, crop2 = sample["crops"]
        assert crop1.shape == crop2.shape
        assert crop1.shape == (3, 224, 224)

    dataloader = dm.train_dataloader()
    for sample in dataloader:
        assert sample["crops"] is not None
        crop1, crop2 = sample["crops"]
        assert crop1.shape == (11, 3, 224, 224)
        assert crop2.shape == (11, 3, 224, 224)
