import torch
from torch import nn
from . import distributed_utils as utils
from collections import OrderedDict


def evaluate(model, data_loader, device, num_classes, criterion):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)

            loss = criterion(output, target)
            total_loss += loss.item()
            num_batches += 1

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    avg_loss = total_loss / max(num_batches, 1)
    return confmat, dice.value.item(), avg_loss


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None, criterion=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    landslide_ratio_sum = 0.0
    num_batches = 0

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        batch_landslide = (target == 1).float().sum().item()
        total_pixels = target.numel()
        landslide_ratio = batch_landslide / total_pixels
        landslide_ratio_sum += landslide_ratio
        num_batches += 1
        if landslide_ratio == 0 and epoch % 5 == 0:
            print(f"⚠️  Warning: No landslide pixels in this batch (epoch {epoch})")

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    avg_landslide_ratio = landslide_ratio_sum / max(num_batches, 1)
    print(f"Epoch {epoch} average landslide pixel ratio: {avg_landslide_ratio:.6f}")

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)