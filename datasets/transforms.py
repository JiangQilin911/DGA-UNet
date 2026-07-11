import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
import torchvision.transforms.functional as TF

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, dem=None):
        for t in self.transforms:
            image, target, dem = t(image, target, dem)
        return image, target, dem

class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target, dem=None):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        if dem is not None:
            dem = F.resize(dem, size, interpolation=T.InterpolationMode.BILINEAR)
        return image, target, dem

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, dem=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
            if dem is not None:
                dem = F.hflip(dem)
        return image, target, dem

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target, dem=None):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
            if dem is not None:
                dem = F.vflip(dem)
        return image, target, dem

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, dem=None):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        if dem is not None:
            dem = pad_if_smaller(dem, self.size, fill=0)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        if dem is not None:
            dem = F.crop(dem, *crop_params)
        return image, target, dem

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, dem=None):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        if dem is not None:
            dem = F.center_crop(dem, self.size)
        return image, target, dem

class ToTensor(object):
    def __call__(self, image, target, dem=None):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        if dem is not None:
            dem = F.to_tensor(dem)
        return image, target, dem

class ColorJitter(object):
    def __init__(self, brightness=0.1, contrast=0.2, saturation=0.2, hue=0.05, p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, image, target, dem=None):
        if random.random() < self.p:
            jitter = T.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue
            )
            image = jitter(image)
        return image, target, dem

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=0.05, p=0.3):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, image, target, dem=None):
        if random.random() < self.p:
            noise = torch.randn_like(image) * self.std + self.mean
            image = image + noise
            image = torch.clamp(image, 0.0, 1.0)
        return image, target, dem

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, dem=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, dem

class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, mean, std,
                 hflip_prob=0.5, vflip_prob=0.5,
                 color_jitter_prob=0.5, brightness=0.1, contrast=0.2, saturation=0.2, hue=0.05,
                 noise_prob=0.3, noise_std=0.05):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        trans = [RandomResize(min_size, max_size)]
        if hflip_prob > 0:
            trans.append(RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(RandomVerticalFlip(vflip_prob))
        trans.append(RandomCrop(crop_size))
        trans.append(ToTensor())

        pixel_trans = []
        if color_jitter_prob > 0:
            pixel_trans.append(ColorJitter(brightness=brightness, contrast=contrast,
                                           saturation=saturation, hue=hue, p=color_jitter_prob))
        if noise_prob > 0:
            pixel_trans.append(GaussianNoise(mean=0.0, std=noise_std, p=noise_prob))

        trans.extend(pixel_trans)
        trans.append(Normalize(mean=mean, std=std))

        self.transforms = Compose(trans)

    def __call__(self, img, target, dem=None):
        return self.transforms(img, target, dem)


class SegmentationPresetEval:
    def __init__(self, mean, std):
        self.transforms = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target, dem=None):
        return self.transforms(img, target, dem)


class SegmentationPresetTest:
    def __init__(self, mean, std):
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.transforms(img)


def get_transform(phase, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 512
    crop_size = 512

    if phase == 'train':
        return SegmentationPresetTrain(
            base_size, crop_size, mean=mean, std=std,
            hflip_prob=0.5, vflip_prob=0.5,
            color_jitter_prob=0.5, brightness=0.1, contrast=0.2, saturation=0.2, hue=0.05,
            noise_prob=0.3, noise_std=0.05
        )
    elif phase == 'val':
        return SegmentationPresetEval(mean=mean, std=std)
    else:   # 'test'
        return SegmentationPresetTest(mean=mean, std=std)