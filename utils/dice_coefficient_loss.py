import torch
import torch.nn as nn
import torch.nn.functional as F

def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    dice_target = target.clone()
    if ignore_index >= 0:
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index
    else:
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    return dice_target.permute(0, 3, 1, 2)

def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    d = 0.
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)
        if ignore_index >= 0:
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]
            t_i = t_i[roi_mask]
        inter = torch.dot(x_i, t_i)
        sets_sum = torch.sum(x_i) + torch.sum(t_i)
        if sets_sum == 0:
            d += 1.0
            continue
        d += (2 * inter + epsilon) / (sets_sum + epsilon)
    return d / batch_size

def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    dice = 0.
    num_valid = 0
    for channel in range(x.shape[1]):
        t_channel = target[:, channel, ...]
        if ignore_index >= 0:
            valid_mask = (t_channel != ignore_index)
            if not valid_mask.any():
                continue
        num_valid += 1
        dice += dice_coeff(x[:, channel, ...], t_channel, ignore_index, epsilon)
    if num_valid == 0:
        return 1.0
    return dice / num_valid

def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    x = nn.functional.softmax(x, dim=1)
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(x, target, ignore_index=ignore_index)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, weight=self.class_weights,
                                  ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.class_weights is None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
                focal_loss = alpha_t * focal_loss
            elif isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[target]
                focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=2, dice_weight=1.0, focal_weight=0.5,
                 ignore_index=255, class_weights=None, focal_alpha=0.25):

        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.focal = FocalLoss(alpha=focal_alpha, gamma=2.0, ignore_index=ignore_index,
                               class_weights=class_weights)

    def forward(self, pred, target):
        dice_target = build_target(target, self.num_classes, self.ignore_index)
        loss_dice = dice_loss(pred, dice_target, multiclass=True, ignore_index=self.ignore_index)
        # Focal loss
        loss_focal = self.focal(pred, target)
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal