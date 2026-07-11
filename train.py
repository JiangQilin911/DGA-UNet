import os
import time
import argparse
import datetime
import random
import torch
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from utils import train_one_epoch, evaluate, create_lr_scheduler
from models import create_model
from datasets import CustomDataset
import torch.backends.cudnn as cudnn
import rasterio as rio
from utils.dice_coefficient_loss import CombinedLoss

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def compute_global_dem_range(dem_files, ignore_value=-9999.0):
    global_min = float('inf')
    global_max = -float('inf')
    for f in dem_files:
        with rio.open(f) as src:
            dem = src.read(1)
            valid = dem[(dem != ignore_value) & np.isfinite(dem) & (dem > -10000) & (dem < 10000)]
            if valid.size > 0:
                global_min = min(global_min, valid.min())
                global_max = max(global_max, valid.max())
    if global_min == float('inf') or global_max == -float('inf'):
        raise ValueError("No valid DEM pixels found.")
    return float(global_min), float(global_max)

def compute_normalization_params(img_files, ignore_value=-9999.0, normalize='zscore'):
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])
    for f in img_files:
        with rio.open(f) as src:
            img = src.read([1,2,3]).astype(np.float32)
        for c in range(3):
            ch = img[c]
            valid = (ch != ignore_value) & np.isfinite(ch)
            if valid.any():
                global_min[c] = min(global_min[c], ch[valid].min())
                global_max[c] = max(global_max[c], ch[valid].max())
    global_min = np.where(np.isinf(global_min), 0.0, global_min)
    global_max = np.where(np.isinf(global_max), 1.0, global_max)

    if normalize == 'zscore':
        sum_vals = np.zeros(3, dtype=np.float64)
        sum_sq   = np.zeros(3, dtype=np.float64)
        total_pixels = 0
        for f in img_files:
            with rio.open(f) as src:
                img = src.read([1,2,3]).astype(np.float32)
            for c in range(3):
                ch = img[c]
                valid = (ch != ignore_value) & np.isfinite(ch)
                if valid.any():
                    sum_vals[c] += ch[valid].sum()
                    sum_sq[c]   += (ch[valid] ** 2).sum()
                    total_pixels += valid.sum()
        if total_pixels == 0:
            mean = [0.372, 0.380, 0.373]
            std  = [0.227, 0.219, 0.223]
        else:
            mean = sum_vals / total_pixels
            var  = sum_sq / total_pixels - mean ** 2
            std  = np.sqrt(np.maximum(var, 1e-8))
        return global_min.tolist(), global_max.tolist(), mean.tolist(), std.tolist()
    else:
        sum_vals = np.zeros(3, dtype=np.float64)
        sum_sq   = np.zeros(3, dtype=np.float64)
        total_pixels = 0
        for f in img_files:
            with rio.open(f) as src:
                img = src.read([1,2,3]).astype(np.float32)
            for c in range(3):
                ch = img[c]
                valid = (ch != ignore_value) & np.isfinite(ch)
                if valid.any():
                    scaled = (ch[valid] - global_min[c]) / (global_max[c] - global_min[c] + 1e-8)
                    sum_vals[c] += scaled.sum()
                    sum_sq[c]   += (scaled ** 2).sum()
                    total_pixels += valid.sum() // 3
        if total_pixels == 0:
            mean = [0.372, 0.380, 0.373]
            std  = [0.227, 0.219, 0.223]
        else:
            mean = sum_vals / total_pixels
            var  = sum_sq / total_pixels - mean ** 2
            std  = np.sqrt(np.maximum(var, 1e-8))
        return global_min.tolist(), global_max.tolist(), mean.tolist(), std.tolist()

def create_model_with_fallback(model_name, num_classes, in_channels, dem_in_channels,
                               encoder_name, encoder_weights):
    try:
        print(f"Creating model with encoder_weights='{encoder_weights}'...")
        model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            in_channels=in_channels,
            dem_in_channels=dem_in_channels,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights
        )
        print("Model created successfully.")
        return model
    except Exception as e:
        if encoder_weights is not None and encoder_weights.lower() != 'none':
            print(f"WARNING: Failed to load pretrained weights: {e}")
            print("Falling back to random initialization.")
            return create_model(
                model_name=model_name,
                num_classes=num_classes,
                in_channels=in_channels,
                dem_in_channels=dem_in_channels,
                encoder_name=encoder_name,
                encoder_weights=None
            )
        else:
            raise e

def save_prediction_comparison(model, val_loader, device, epoch, save_dir, num_samples=3):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    with torch.no_grad():
        for image, target in val_loader:
            if count >= num_samples:
                break
            image, target = image.to(device), target.to(device)
            output = model(image)
            pred = output.argmax(1)
            for i in range(image.size(0)):
                if count >= num_samples:
                    break
                if target[i].sum() == 0 and count < num_samples - 1:
                    continue
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(image[i, 0].cpu(), cmap='gray')
                axes[0].set_title('Input (Slope)')
                axes[0].axis('off')
                # 真值
                axes[1].imshow(target[i].cpu(), cmap='gray')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                # 预测
                axes[2].imshow(pred[i].cpu(), cmap='gray')
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                error_map = np.zeros((target[i].shape[0], target[i].shape[1], 3), dtype=np.uint8)
                gt = target[i].cpu().numpy()
                pr = pred[i].cpu().numpy()
                error_map[(gt == 1) & (pr == 0)] = [0, 0, 255]
                error_map[(gt == 0) & (pr == 1)] = [255, 0, 0]
                error_map[(gt == 1) & (pr == 1)] = [0, 255, 0]
                axes[3].imshow(error_map)
                axes[3].set_title('Errors (Red:FP, Blue:FN, Green:TP)')
                axes[3].axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_sample_{count}.png'), dpi=100)
                plt.close()
                count += 1
    model.train()

def save_confusion_matrix_heatmap(confmat, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    mat = confmat.mat.cpu().numpy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Background', 'Landslide'],
                yticklabels=['Background', 'Landslide'])
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'confmat_epoch_{epoch}.png'), dpi=100)
    plt.close()

def compute_model_params_and_speed(model, device, input_size=(1, 7, 256, 256)):
    model.eval()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_all = sum(p.numel() for p in model.parameters())
    dummy_input = torch.randn(input_size).to(device)

    # 预热
    for _ in range(10):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    # 测时
    times = []
    with torch.no_grad():
        for _ in range(50):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    avg_time = np.mean(times) * 1000  # ms

    return total_params, total_params_all, avg_time

def save_convergence_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(epochs, history['val_loss'], 'c--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Dice', color='tab:red')
    ax2.plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    plt.title('Training Convergence')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main(args):
    seed_everything(args.seed)
    os.makedirs(args.weight_path, exist_ok=True)
    results_file = os.path.join(args.weight_path,
                                f"results_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_classes = args.num_classes

    tb_writer = SummaryWriter(log_dir=os.path.join(args.weight_path, 'tensorboard'))

    if torch.cuda.is_available():
        cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    train_root = os.path.join(args.data_path, 'train')
    val_root = os.path.join(args.data_path, 'val')
    if not os.path.exists(train_root) or not os.path.exists(val_root):
        raise FileNotFoundError(f"Train/Val folders not found under {args.data_path}")

    dem_train_list, img_train_list, mask_train_list = [], [], []
    for img_file in sorted(glob(os.path.join(train_root, 'images', '*.tif'))):
        base = os.path.basename(img_file)
        dem_file = os.path.join(train_root, 'dems', base)
        mask_file = os.path.join(train_root, 'masks', base)
        if not os.path.exists(dem_file):
            raise FileNotFoundError(f"DEM missing for {img_file}")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask missing for {img_file}")
        dem_train_list.append(dem_file)
        img_train_list.append(img_file)
        mask_train_list.append(mask_file)

    if args.dem_min is None or args.dem_max is None:
        print("Computing global DEM min/max from training set...")
        dem_min, dem_max = compute_global_dem_range(dem_train_list, ignore_value=-9999.0)
        print(f"Global DEM min: {dem_min:.4f}, max: {dem_max:.4f}")
        args.dem_min = dem_min
        args.dem_max = dem_max
    else:
        dem_min, dem_max = args.dem_min, args.dem_max

    if (args.rgb_min is None or args.rgb_max is None or
        args.rgb_mean is None or args.rgb_std is None):
        print("Computing global feature normalization...")
        rgb_min, rgb_max, feat_mean, feat_std = compute_normalization_params(
            img_train_list, ignore_value=-9999.0, normalize=args.rgb_normalize
        )
        args.rgb_min, args.rgb_max = rgb_min, rgb_max
        args.rgb_mean, args.rgb_std = feat_mean, feat_std
        print(f"Feature min: {rgb_min}, max: {rgb_max}")
        print(f"Feature mean: {feat_mean}, std: {feat_std}")
    else:
        rgb_min, rgb_max = args.rgb_min, args.rgb_max

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(
        root=args.data_path, train=True, transforms=train_transform,
        dem_folder='dems', dem_mean=args.dem_mean, dem_std=args.dem_std,
        dem_min=dem_min, dem_max=dem_max,
        rgb_min=rgb_min, rgb_max=rgb_max,
        rgb_mean=args.rgb_mean, rgb_std=args.rgb_std,
        rgb_normalize=args.rgb_normalize
    )
    val_dataset = CustomDataset(
        root=args.data_path, train=False, transforms=val_transform,
        dem_folder='dems', dem_mean=args.dem_mean, dem_std=args.dem_std,
        dem_min=dem_min, dem_max=dem_max,
        rgb_min=rgb_min, rgb_max=rgb_max,
        rgb_mean=args.rgb_mean, rgb_std=args.rgb_std,
        rgb_normalize=args.rgb_normalize
    )

    num_workers = min(os.cpu_count() or 1, batch_size, 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, pin_memory=True, persistent_workers=num_workers>0)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=num_workers,
                            pin_memory=True, persistent_workers=num_workers>0)

    model = create_model_with_fallback(
        model_name=args.model_name,
        num_classes=num_classes,
        in_channels=7,
        dem_in_channels=4,
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights
    )
    model.to(device)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == 'cuda' else None
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True)

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if scaler and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.class_weights is not None:
        class_weights = torch.tensor(args.class_weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    criterion = CombinedLoss(
        num_classes=num_classes,
        dice_weight=args.dice_weight,
        focal_weight=args.focal_weight,
        ignore_index=255,
        class_weights=class_weights,
        focal_alpha=args.focal_alpha
    ).to(device)

    print("\n=== Final Training Configuration ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print(f"Input channels: 7, DEM branch channels: 4")
    print(f"AMP: {'ON' if scaler else 'OFF'}")
    print(f"Loss: dice_weight={args.dice_weight}, focal_weight={args.focal_weight}, focal_alpha={args.focal_alpha}, class_weights={class_weights}")
    print("=====================================\n")

    with open(results_file, "a") as f:
        f.write("=== Training Parameters ===\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write("Input channels: 7, DEM branch channels: 4\n")
        f.write("====================================\n\n")

    history = {'train_loss': [], 'val_dice': [], 'val_loss': []}

    best_dice = 0.0
    patience_counter = 0
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(
            model, optimizer, train_loader, device, epoch, num_classes,
            lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler,
            criterion=criterion
        )
        confmat, landslide_dice, val_loss = evaluate(model, val_loader, device, num_classes, criterion)
        val_info = str(confmat)
        print(val_info)
        print(f"Landslide Dice: {landslide_dice:.4f}, Val Loss: {val_loss:.4f}")

        tb_writer.add_scalar('Loss/train', mean_loss, epoch)
        tb_writer.add_scalar('Loss/val', val_loss, epoch)
        tb_writer.add_scalar('Dice/val', landslide_dice, epoch)
        tb_writer.add_scalar('LR', lr, epoch)

        history['train_loss'].append(mean_loss)
        history['val_dice'].append(landslide_dice)
        history['val_loss'].append(val_loss)

        with open(results_file, "a") as f:
            f.write(f"[epoch: {epoch}]\ntrain_loss: {mean_loss:.4f}\nval_loss: {val_loss:.4f}\nlr: {lr:.6f}\nlandslide_dice: {landslide_dice:.4f}\n{val_info}\n\n")

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            vis_dir = os.path.join(args.weight_path, 'visuals')
            save_prediction_comparison(model, val_loader, device, epoch, vis_dir)
            save_confusion_matrix_heatmap(confmat, epoch, vis_dir)

        if landslide_dice > best_dice:
            best_dice = landslide_dice
            patience_counter = 0
            save_file = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
                "dem_min": dem_min, "dem_max": dem_max,
                "dem_mean": args.dem_mean, "dem_std": args.dem_std,
                "rgb_min": rgb_min, "rgb_max": rgb_max,
                "rgb_mean": args.rgb_mean, "rgb_std": args.rgb_std,
            }
            if scaler:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, os.path.join(args.weight_path, "best_model.pth"))
            print(f"Saved best model (landslide dice={best_dice:.4f}) at epoch {epoch}")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"Training finished in {datetime.timedelta(seconds=int(total_time))}")

    curve_path = os.path.join(args.weight_path, 'convergence_curve.png')
    save_convergence_curves(history, curve_path)
    print(f"Convergence curve saved to {curve_path}")

    total_params, total_params_all, avg_infer_time = compute_model_params_and_speed(model, device)
    speed_info = (
        f"Model: {args.model_name}\n"
        f"Trainable params: {total_params / 1e6:.2f} M\n"
        f"Total params: {total_params_all / 1e6:.2f} M\n"
        f"Average inference time (256x256): {avg_infer_time:.2f} ms\n"
    )
    print(speed_info)
    with open(results_file, "a") as f:
        f.write("\n" + speed_info)
    with open(os.path.join(args.weight_path, "model_speed.txt"), "w") as f:
        f.write(speed_info)

    tb_writer.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Dual-branch segmentation training with DEM")
    parser.add_argument("--data-path", default="./data")
    parser.add_argument("--weight-path", default="./work_dirs/exp")
    parser.add_argument("--model-name", type=str, default="dual_unet",
                        choices=['dual_unet','dual_segformer','dual_deeplabv3p','dual_pspnet','dual_fpn','dual_linknet'])
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("-b", "--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate (reduced for new channels)")
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--print-freq", default=1, type=int)
    parser.add_argument("--resume", default="")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--dem-mean", type=float, default=0.5)
    parser.add_argument("--dem-std", type=float, default=0.2)
    parser.add_argument("--dem-min", type=float, default=None)
    parser.add_argument("--dem-max", type=float, default=None)
    parser.add_argument("--rgb-mean", type=float, nargs=3, default=None)
    parser.add_argument("--rgb-std", type=float, nargs=3, default=None)
    parser.add_argument("--rgb-min", type=float, nargs=3, default=None)
    parser.add_argument("--rgb-max", type=float, nargs=3, default=None)
    parser.add_argument("--rgb-normalize", type=str, default="zscore", choices=["minmax", "zscore"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--encoder-name", type=str, default="resnet50")
    parser.add_argument("--encoder-weights", type=str, default=None)

    parser.add_argument("--dice-weight", type=float, default=1.0, help="Weight for Dice loss")
    parser.add_argument("--focal-weight", type=float, default=0.5, help="Weight for Focal loss")
    parser.add_argument("--focal-alpha", type=float, default=0.25,
                        help="Alpha parameter for Focal loss (ignored if class_weights provided)")
    parser.add_argument("--class-weights", type=float, nargs=2, default=[1.0, 10.0],
                        help="Class weights for background and landslide, e.g., '1.0 10.0'. If not provided, no class weights.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)