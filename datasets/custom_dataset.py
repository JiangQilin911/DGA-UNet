import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import rasterio
import cv2
from skimage.transform import resize


class CustomDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None,
                 dem_folder=None, dem_mean=0.5, dem_std=0.2,
                 dem_min=None, dem_max=None,
                 rgb_min=None, rgb_max=None,
                 rgb_mean=None, rgb_std=None,
                 rgb_normalize='minmax'):
        super().__init__()
        self.flag = "train" if train else "val"
        data_root = os.path.join(root, self.flag)
        assert os.path.exists(data_root), f"Path '{data_root}' does not exist."

        self.transforms = transforms
        self.img_list = sorted(glob(os.path.join(data_root, "images", "*.tif")))
        self.mask_list = sorted(glob(os.path.join(data_root, "masks", "*.tif")))
        assert len(self.img_list) == len(self.mask_list), "Images and masks count mismatch"

        self.dem_folder = dem_folder
        self.dem_mean = dem_mean
        self.dem_std = dem_std
        self.dem_min = dem_min
        self.dem_max = dem_max
        if self.dem_folder is not None:
            if not os.path.isabs(self.dem_folder):
                self.dem_folder = os.path.join(root, self.flag, self.dem_folder)
            assert os.path.exists(self.dem_folder), f"DEM folder '{self.dem_folder}' not found."

        self.rgb_normalize = rgb_normalize
        if rgb_normalize == 'minmax':
            self.rgb_min = np.array(rgb_min, dtype=np.float32) if rgb_min is not None else None
            self.rgb_max = np.array(rgb_max, dtype=np.float32) if rgb_max is not None else None
            self.rgb_mean = None
            self.rgb_std = None
        elif rgb_normalize == 'zscore':
            self.rgb_mean = np.array(rgb_mean, dtype=np.float32) if rgb_mean is not None else None
            self.rgb_std = np.array(rgb_std, dtype=np.float32) if rgb_std is not None else None
            self.rgb_min = None
            self.rgb_max = None
        else:
            raise ValueError("rgb_normalize must be 'minmax' or 'zscore'")

    def _normalize_rgb(self, arr):
        if self.rgb_normalize == 'minmax':
            if self.rgb_min is not None and self.rgb_max is not None:
                arr = (arr - self.rgb_min) / (self.rgb_max - self.rgb_min + 1e-8)
                arr = np.clip(arr, 0, 1)
            else:
                minv, maxv = arr.min(), arr.max()
                if maxv - minv > 0:
                    arr = (arr - minv) / (maxv - minv)
                else:
                    arr = np.zeros_like(arr)
        else:  # zscore
            if self.rgb_mean is not None and self.rgb_std is not None:
                arr = (arr - self.rgb_mean) / (self.rgb_std + 1e-8)
                arr = np.clip(arr, -3, 3)
                arr = (arr + 3) / 6
            else:
                minv, maxv = arr.min(), arr.max()
                if maxv - minv > 0:
                    arr = (arr - minv) / (maxv - minv)
                else:
                    arr = np.zeros_like(arr)
        return (arr * 255).astype(np.uint8)

    def __getitem__(self, idx):
        with rasterio.open(self.img_list[idx]) as src:
            img_arr = src.read().astype(np.float32)
        img_arr = np.where((img_arr > -10000) & (img_arr < 10000), img_arr, np.nan)
        if self.rgb_normalize == 'minmax' and self.rgb_min is not None:
            fill_value = self.rgb_min[:, np.newaxis, np.newaxis]
        else:
            fill_value = 0.0
        img_arr = np.nan_to_num(img_arr, nan=fill_value)
        img_arr = np.transpose(img_arr, (1, 2, 0))
        img_uint8 = self._normalize_rgb(img_arr)

        with rasterio.open(self.mask_list[idx]) as src:
            mask_arr = src.read(1).astype(np.int32)

        dem_multi = None
        if self.dem_folder is not None:
            dem_filename = os.path.basename(self.img_list[idx])
            dem_path = os.path.join(self.dem_folder, dem_filename)
            if os.path.exists(dem_path):
                with rasterio.open(dem_path) as src:
                    dem_raw = src.read(1).astype(np.float64)
                dem_raw = np.where(dem_raw < -9000, np.nan, dem_raw)
                if self.dem_min is not None:
                    dem_raw = np.nan_to_num(dem_raw, nan=self.dem_min)
                else:
                    dem_raw = np.nan_to_num(dem_raw, nan=0.0)
                dem_arr = dem_raw.astype(np.float32)

                if dem_arr.shape != img_uint8.shape[:2]:
                    dem_arr = resize(dem_arr, img_uint8.shape[:2],
                                     preserve_range=True, anti_aliasing=False)

                dem_smooth = cv2.GaussianBlur(dem_arr, (3, 3), 0)
                sobel_x = cv2.Sobel(dem_smooth, cv2.CV_32F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(dem_smooth, cv2.CV_32F, 0, 1, ksize=3)

                grad_mag = np.sqrt(sobel_x**2 + sobel_y**2 + 1e-8)
                grad_mag = grad_mag / (grad_mag.max() + 1e-8)

                grad_angle = np.arctan2(sobel_y, sobel_x)
                grad_sin = (np.sin(grad_angle) + 1) / 2
                grad_cos = (np.cos(grad_angle) + 1) / 2

                if self.dem_min is not None and self.dem_max is not None:
                    dem_norm = (dem_arr - self.dem_min) / (self.dem_max - self.dem_min + 1e-8)
                    dem_norm = np.clip(dem_norm, 0, 1)
                else:
                    minv, maxv = dem_arr.min(), dem_arr.max()
                    if maxv - minv > 0:
                        dem_norm = (dem_arr - minv) / (maxv - minv)
                    else:
                        dem_norm = np.zeros_like(dem_arr)
                dem_norm = (dem_norm - self.dem_mean) / (self.dem_std + 1e-8)

                dem_multi = np.stack([dem_norm, grad_mag, grad_sin, grad_cos], axis=-1)
            else:
                dem_multi = np.zeros((img_uint8.shape[0], img_uint8.shape[1], 4), dtype=np.float32)

        if self.flag == 'train':
            if random.random() > 0.5:
                img_uint8 = np.fliplr(img_uint8)
                mask_arr = np.fliplr(mask_arr)
                if dem_multi is not None:
                    dem_multi = np.fliplr(dem_multi)
            if random.random() > 0.5:
                img_uint8 = np.flipud(img_uint8)
                mask_arr = np.flipud(mask_arr)
                if dem_multi is not None:
                    dem_multi = np.flipud(dem_multi)

        img_pil = Image.fromarray(img_uint8)
        mask_pil = Image.fromarray(mask_arr)

        if self.transforms:
            img_tensor = self.transforms(img_pil)
        else:
            img_tensor = torch.from_numpy(np.array(img_pil).transpose(2,0,1)).float() / 255.0

        mask_tensor = torch.as_tensor(np.array(mask_pil), dtype=torch.long)

        if dem_multi is not None:
            dem_tensor = torch.from_numpy(dem_multi.transpose(2, 0, 1).copy()).float()
            img_tensor = torch.cat([img_tensor, dem_tensor], dim=0)

        return img_tensor, mask_tensor

    def __len__(self):
        return len(self.img_list)


class TestDataset(Dataset):
    def __init__(self, root: str, transforms=None):
        super().__init__()
        assert os.path.exists(root), f"Path '{root}' does not exist."
        self.transforms = transforms
        self.img_list = sorted(glob(os.path.join(root, "*.tif")))

    def __getitem__(self, idx):
        with rasterio.open(self.img_list[idx]) as src:
            img_arr = src.read()
        img_arr = np.transpose(img_arr, (1,2,0))
        minv, maxv = img_arr.min(), img_arr.max()
        if maxv - minv > 0:
            img_arr = ((img_arr - minv) / (maxv - minv) * 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        img_name = os.path.basename(self.img_list[idx])
        if self.transforms:
            img = self.transforms(img)
        return img, img_name

    def __len__(self):
        return len(self.img_list)