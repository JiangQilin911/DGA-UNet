import os
import argparse
import numpy as np
import torch
import rasterio as rio
from glob import glob
from tqdm import tqdm
import cv2
from skimage.transform import resize

from models import create_model


def process_dem(dem_raw, dem_min, dem_max, dem_mean=0.5, dem_std=0.2, target_shape=None):
    dem_raw = np.where(dem_raw < -9000, np.nan, dem_raw)
    dem_raw = np.nan_to_num(dem_raw, nan=dem_min if dem_min is not None else 0.0)
    dem_arr = dem_raw.astype(np.float32)

    if target_shape is not None and dem_arr.shape != target_shape:
        dem_arr = resize(dem_arr, target_shape, preserve_range=True, anti_aliasing=False)

    dem_smooth = cv2.GaussianBlur(dem_arr, (3, 3), 0)
    sobel_x = cv2.Sobel(dem_smooth, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(dem_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2 + 1e-8)
    if grad_mag.max() > 0:
        grad_mag = grad_mag / grad_mag.max()
    grad_angle = np.arctan2(sobel_y, sobel_x)
    grad_sin = (np.sin(grad_angle) + 1) / 2
    grad_cos = (np.cos(grad_angle) + 1) / 2

    if dem_min is not None and dem_max is not None:
        dem_norm = (dem_arr - dem_min) / (dem_max - dem_min + 1e-8)
        dem_norm = np.clip(dem_norm, 0, 1)
    else:
        minv, maxv = dem_arr.min(), dem_arr.max()
        dem_norm = (dem_arr - minv) / (maxv - minv + 1e-8) if maxv > minv else np.zeros_like(dem_arr)
    dem_norm = (dem_norm - dem_mean) / (dem_std + 1e-8)

    dem_multi = np.stack([dem_norm, grad_mag, grad_sin, grad_cos], axis=-1)
    return dem_multi.astype(np.float32)


def slide_infer(img_7ch, model, win_size, step_size, device):
    _, H, W = img_7ch.shape
    pad_h = (win_size - H % win_size) % win_size
    pad_w = (win_size - W % win_size) % win_size
    img_pad = torch.nn.functional.pad(img_7ch, (0, pad_w, 0, pad_h), mode='constant', value=0)
    _, Hp, Wp = img_pad.shape
    mask = np.zeros((Hp, Wp), dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for y in range(0, Hp, step_size):
            for x in range(0, Wp, step_size):
                if y + win_size > Hp:
                    y = Hp - win_size
                if x + win_size > Wp:
                    x = Wp - win_size
                win = img_pad[:, y:y+win_size, x:x+win_size].unsqueeze(0).to(device)
                pred = model(win).argmax(1).squeeze(0).cpu().numpy()
                mask[y:y+win_size, x:x+win_size] = pred
    return mask[:H, :W]


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    train_args = checkpoint['args']

    dem_min = checkpoint['dem_min']
    dem_max = checkpoint['dem_max']
    dem_mean = checkpoint.get('dem_mean', 0.5)
    dem_std = checkpoint.get('dem_std', 0.2)

    rgb_normalize = getattr(train_args, 'rgb_normalize', 'minmax')
    if rgb_normalize == 'minmax':
        rgb_min = checkpoint['rgb_min']
        rgb_max = checkpoint['rgb_max']
    elif rgb_normalize == 'zscore':
        rgb_mean = checkpoint['rgb_mean']
        rgb_std = checkpoint['rgb_std']
    else:
        raise ValueError(f"Unknown rgb_normalize: {rgb_normalize}")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    model = create_model(
        model_name=train_args.model_name,
        num_classes=train_args.num_classes,
        in_channels=7,
        dem_in_channels=4,
        encoder_name=train_args.encoder_name,
        encoder_weights=None
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()

    img_files = sorted(glob(os.path.join(args.img_path, '*.tif')))
    os.makedirs(args.out_path, exist_ok=True)

    for img_file in tqdm(img_files, desc="Predicting"):
        base = os.path.basename(img_file)
        dem_file = os.path.join(args.dem_path, base)
        if not os.path.exists(dem_file):
            raise FileNotFoundError(f"DEM file not found: {dem_file}")

        with rio.open(img_file) as src:
            img = src.read([1, 2, 3]).astype(np.float32)
            meta = src.meta

        with rio.open(dem_file) as src:
            dem = src.read(1).astype(np.float32)

        dem_multi = process_dem(dem, dem_min, dem_max, dem_mean, dem_std,
                                target_shape=img.shape[1:])

        if rgb_normalize == 'minmax':
            for c in range(3):
                img[c] = (img[c] - rgb_min[c]) / (rgb_max[c] - rgb_min[c] + 1e-8)
                img[c] = np.clip(img[c], 0, 1)
        else:
            for c in range(3):
                img[c] = (img[c] - rgb_mean[c]) / (rgb_std[c] + 1e-8)
                img[c] = np.clip(img[c], -3, 3)
                img[c] = (img[c] + 3) / 6

        for c in range(3):
            img[c] = (img[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

        img_7ch = np.concatenate([img, dem_multi.transpose(2, 0, 1)], axis=0)
        img_tensor = torch.from_numpy(img_7ch).float()

        mask = slide_infer(img_tensor, model, args.win_size, args.step_size, device)

        out_meta = meta.copy()
        out_meta.update(count=1, dtype='uint8', nodata=0, compress='lzw')
        out_file = os.path.join(args.out_path, base)
        with rio.open(out_file, 'w', **out_meta) as dst:
            dst.write(mask, 1)


if __name__ == "__main__":

    WEIGHT_PATH = r"D:\geoseg\work_dirs\exp\best_model.pth"
    IMG_PATH   = r"D:\geoseg\data\prediction\images"
    DEM_PATH   = r"D:\geoseg\data\prediction\dems"
    OUT_PATH   = r"D:\geoseg\data\prediction\results"
    WIN_SIZE   = 256
    STEP_SIZE  = 256

    args = argparse.Namespace(
        weight_path=WEIGHT_PATH,
        img_path=IMG_PATH,
        dem_path=DEM_PATH,
        out_path=OUT_PATH,
        win_size=WIN_SIZE,
        step_size=STEP_SIZE
    )
    main(args)