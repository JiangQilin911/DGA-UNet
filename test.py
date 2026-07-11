import os
import argparse
import random
import torch
import numpy as np
import rasterio as rio
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from torch.utils.data import Dataset, DataLoader
from models import create_model


class TestDataset(Dataset):
    def __init__(self, img_dir, dem_dir,
                 rgb_min, rgb_max, rgb_mean, rgb_std,
                 dem_min, dem_max, dem_mean, dem_std):
        self.img_files = sorted(glob(os.path.join(img_dir, '*.tif')))
        if not self.img_files:
            raise FileNotFoundError(f"No images in {img_dir}")
        self.dem_files = []
        for f in self.img_files:
            base = os.path.basename(f)
            dem_f = os.path.join(dem_dir, base)
            if not os.path.exists(dem_f):
                raise FileNotFoundError(f"DEM not found: {dem_f}")
            self.dem_files.append(dem_f)
        self.rgb_min = np.array(rgb_min, dtype=np.float32)
        self.rgb_max = np.array(rgb_max, dtype=np.float32)
        self.rgb_mean = np.array(rgb_mean, dtype=np.float32)
        self.rgb_std = np.array(rgb_std, dtype=np.float32)
        self.dem_min = dem_min
        self.dem_max = dem_max
        self.dem_mean = dem_mean
        self.dem_std = dem_std

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        dem_path = self.dem_files[idx]
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        with rio.open(img_path) as src:
            rgb = src.read([1,2,3]).astype(np.float32)

        with rio.open(dem_path) as src:
            dem_raw = src.read(1).astype(np.float64)
        dem_raw = np.where(dem_raw < -9000, np.nan, dem_raw)
        dem = np.nan_to_num(dem_raw, nan=self.dem_min).astype(np.float32)

        rgb = np.where((rgb > -10000) & (rgb < 10000), rgb, np.nan)
        rgb = np.nan_to_num(rgb, nan=self.rgb_min[:, np.newaxis, np.newaxis])

        rgb = (rgb - self.rgb_min[:, None, None]) / (self.rgb_max[:, None, None] - self.rgb_min[:, None, None] + 1e-8)
        rgb = np.clip(rgb, 0, 1)
        rgb = (rgb - self.rgb_mean[:, None, None]) / (self.rgb_std[:, None, None])

        if self.dem_max - self.dem_min > 1e-6:
            dem_norm = (dem - self.dem_min) / (self.dem_max - self.dem_min + 1e-8)
        else:
            dem_norm = np.zeros_like(dem)
        dem_norm = np.clip(dem_norm, 0, 1)
        dem_norm = (dem_norm - self.dem_mean) / (self.dem_std + 1e-8)

        dem_smooth = cv2.GaussianBlur(dem, (3, 3), 0)
        sobel_x = cv2.Sobel(dem_smooth, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(dem_smooth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2 + 1e-8)
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)
        grad_angle = np.arctan2(sobel_y, sobel_x)
        grad_sin = (np.sin(grad_angle) + 1) / 2
        grad_cos = (np.cos(grad_angle) + 1) / 2

        dem_multi = np.stack([dem_norm, grad_mag, grad_sin, grad_cos], axis=0)

        img = np.concatenate([rgb, dem_multi], axis=0).astype(np.float32)

        return torch.from_numpy(img), base_name, img_path


def compute_metrics(conf_mat):
    ious, dices = [], []
    for i in range(conf_mat.shape[0]):
        tp = conf_mat[i,i]
        fp = conf_mat[:,i].sum() - tp
        fn = conf_mat[i,:].sum() - tp
        iou = tp / (tp+fp+fn) if (tp+fp+fn) > 0 else 0
        dice = 2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
        ious.append(iou)
        dices.append(dice)
    return ious, dices


def plot_confusion_matrix(conf, save_path):
    plt.figure(figsize=(6,5))
    sns.heatmap(conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background','Landslide'],
                yticklabels=['Background','Landslide'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix heatmap saved to {save_path}")


def plot_sample_comparisons(img_dir, label_dir, pred_dir, sample_names, save_dir, num_samples=5):
    if len(sample_names) == 0:
        return
    selected = random.sample(sample_names, min(num_samples, len(sample_names)))
    for name in selected:
        img_path = os.path.join(img_dir, name + '.tif')
        if not os.path.exists(img_path):
            continue
        with rio.open(img_path) as src:
            slope = src.read(1).astype(np.float32)
        s_min, s_max = slope.min(), slope.max()
        if s_max - s_min > 0:
            slope_disp = (slope - s_min) / (s_max - s_min)
        else:
            slope_disp = np.zeros_like(slope)

        label_path = os.path.join(label_dir, name + '.tif')
        if not os.path.exists(label_path):
            continue
        with rio.open(label_path) as src:
            label = src.read(1)

        pred_path = os.path.join(pred_dir, name + '.tif')
        if not os.path.exists(pred_path):
            continue
        with rio.open(pred_path) as src:
            pred = src.read(1)

        error_map = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
        error_map[(label == 1) & (pred == 0)] = [0, 0, 255]
        error_map[(label == 0) & (pred == 1)] = [255, 0, 0]
        error_map[(label == 1) & (pred == 1)] = [0, 255, 0]

        fig, axes = plt.subplots(1, 4, figsize=(20,5))
        axes[0].imshow(slope_disp, cmap='gray')
        axes[0].set_title('Input (Slope)')
        axes[0].axis('off')
        axes[1].imshow(label, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        axes[2].imshow(pred, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        axes[3].imshow(error_map)
        axes[3].set_title('Errors (Red:FP, Blue:FN, Green:TP)')
        axes[3].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'sample_{name}.png'), dpi=150)
        plt.close()
    print(f"Sample comparison images saved to {save_dir}")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.weight_path}")
    ckpt = torch.load(args.weight_path, map_location='cpu', weights_only=False)

    rgb_min = ckpt.get('rgb_min', args.rgb_min)
    rgb_max = ckpt.get('rgb_max', args.rgb_max)
    rgb_mean = ckpt.get('rgb_mean', args.rgb_mean)
    rgb_std = ckpt.get('rgb_std', args.rgb_std)
    dem_min = ckpt.get('dem_min', args.dem_min)
    dem_max = ckpt.get('dem_max', args.dem_max)
    dem_mean = ckpt.get('dem_mean', args.dem_mean)
    dem_std = ckpt.get('dem_std', args.dem_std)

    if 'args' in ckpt:
        encoder_name = getattr(ckpt['args'], 'encoder_name', args.encoder_name)
    else:
        encoder_name = args.encoder_name

    if 'args' in ckpt and hasattr(ckpt['args'], 'num_classes'):
        num_classes = ckpt['args'].num_classes
    else:
        num_classes = args.num_classes

    if dem_min is None or dem_max is None:
        raise ValueError("DEM min/max not found in checkpoint or arguments.")
    if rgb_min is None or rgb_max is None:
        raise ValueError("RGB min/max not found in checkpoint or arguments.")

    model = create_model(
        args.model_name, num_classes, in_channels=7, dem_in_channels=4,
        encoder_name=encoder_name,
        encoder_weights=None
    )
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    if args.test_dir:
        test_root = args.test_dir
        img_dir = os.path.join(test_root, 'images')
        dem_dir = os.path.join(test_root, 'dems')
    else:
        test_root = os.path.join(args.data_path, 'test')
        img_dir = os.path.join(test_root, 'images')
        dem_dir = os.path.join(test_root, 'dems')
    if not os.path.exists(img_dir) or not os.path.exists(dem_dir):
        raise FileNotFoundError(f"test directory must contain images/ and dems/ subfolders")

    dataset = TestDataset(img_dir, dem_dir,
                          rgb_min, rgb_max, rgb_mean, rgb_std,
                          dem_min, dem_max, dem_mean, dem_std)
    loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)

    os.makedirs(args.out_path, exist_ok=True)

    sample_names = []
    with torch.no_grad():
        for img, img_name, src_path in tqdm(loader, desc="Inference"):
            img = img.to(device)
            if args.tta:
                out1 = model(img)
                prob1 = torch.softmax(out1, dim=1)
                img_flip = torch.flip(img, dims=[3])
                out2 = model(img_flip)
                prob2 = torch.softmax(torch.flip(out2, dims=[3]), dim=1)
                prob = (prob1 + prob2) / 2.0
            else:
                out = model(img)
                prob = torch.softmax(out, dim=1)
            pred = prob.argmax(1).squeeze().cpu().numpy().astype(np.uint8)

            with rio.open(src_path[0]) as src:
                profile = src.profile.copy()
            profile.update(driver='GTiff', count=1, dtype='uint8', compress='lzw', nodata=None)
            out_file = os.path.join(args.out_path, img_name[0] + '.tif')
            with rio.open(out_file, 'w', **profile) as dst:
                dst.write(pred, 1)
            sample_names.append(img_name[0])

    if args.label_dir:
        label_files = glob(os.path.join(args.label_dir, '*.tif'))
        if not label_files:
            print("No label files found in", args.label_dir)
            return
        conf = np.zeros((num_classes, num_classes), dtype=np.int64)
        all_preds, all_labels = [], []
        for lab_file in tqdm(label_files, desc="Evaluating"):
            name = os.path.basename(lab_file)
            out_file = os.path.join(args.out_path, name)
            if not os.path.exists(out_file):
                continue
            with rio.open(lab_file) as src:
                lab = src.read(1).flatten()
            with rio.open(out_file) as src:
                pred = src.read(1).flatten()
            conf += confusion_matrix(lab, pred, labels=list(range(num_classes)))
            all_labels.append(lab)
            all_preds.append(pred)

        ious, dices = compute_metrics(conf)
        print("\nConfusion Matrix:")
        print(conf)
        for i in range(num_classes):
            print(f"Class {i}: IoU={ious[i]:.4f}, Dice={dices[i]:.4f}")
        if all_labels:
            kappa = cohen_kappa_score(np.concatenate(all_labels), np.concatenate(all_preds))
            print(f"Kappa: {kappa:.4f}")

        vis_dir = os.path.join(args.out_path, 'visuals')
        os.makedirs(vis_dir, exist_ok=True)

        plot_confusion_matrix(conf, os.path.join(vis_dir, 'confusion_matrix.png'))

        sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in label_files]
        plot_sample_comparisons(img_dir, args.label_dir, args.out_path,
                                sample_ids, vis_dir, num_samples=5)


def parse_args():
    parser = argparse.ArgumentParser(description="Test dual-branch model")
    parser.add_argument("--weight-path", type=str, default="./work_dirs/exp/best_model.pth")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--test-dir", type=str, default=None)
    parser.add_argument("--out-path", type=str, default="./test_results")
    parser.add_argument("--label-dir", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="dual_unet",
                        choices=['dual_unet','dual_segformer','dual_deeplabv3p','dual_pspnet','dual_fpn','dual_linknet'])
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tta", action="store_true", help="Test-time augmentation")
    parser.add_argument("--encoder-name", type=str, default="resnet50")
    parser.add_argument("--dem-mean", type=float, default=0.5)
    parser.add_argument("--dem-std", type=float, default=0.2)
    parser.add_argument("--dem-min", type=float, default=None)
    parser.add_argument("--dem-max", type=float, default=None)
    parser.add_argument("--rgb-mean", type=float, nargs=3, default=[0.372,0.380,0.373])
    parser.add_argument("--rgb-std", type=float, nargs=3, default=[0.227,0.219,0.223])
    parser.add_argument("--rgb-min", type=float, nargs=3, default=None)
    parser.add_argument("--rgb-max", type=float, nargs=3, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("=== Test Configuration ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    main(args)