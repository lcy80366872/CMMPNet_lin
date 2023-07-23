"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import cv2
import numpy as np

import random

from PIL import Image, ImageEnhance


def randomHueSaturationValue(
    image,
    hue_shift_limit=(-30, 30),
    sat_shift_limit=(-5, 5),
    val_shift_limit=(-15, 15),
    u=0.5,
):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(
            hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(
    image,
    mask,
    shift_limit=(-0.1, 0.1),
    scale_limit=(-0.1, 0.1),
    aspect_limit=(-0.1, 0.1),
    rotate_limit=(-0, 0),
    borderMode=cv2.BORDER_CONSTANT,
    u=0.5,
):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy]
        )

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(
            image,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpPerspective(
            mask,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotation(image, label, mode=Image.BICUBIC):
    """
     对图像进行随机任意角度(0~360度)旋转
    :param mode 邻近插值,双线性插值,双三次B样条插值(default)
    :param image PIL的图像image
    :return: 旋转转之后的图像
    """
    random_angle = np.random.randint(1, 360)
    return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

def randomColor(image, label):
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label  # 调整图像锐度

def randomGaussian(image, label, mean=0.2, sigma=0.3,u=0.1):
    """
     对图像进行高斯噪声处理
    :param image:
    :return:
    """

    def gaussianNoisy(im, mean=0.2, sigma=0.3):
        """
        对图像做高斯噪音处理
        :param im: 单通道图像
        :param mean: 偏移量
        :param sigma: 标准差
        :return:
        """
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    # 将 图像转化成数组
    if np.random.random() < u:
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return img, label
    else:
        return image,label

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask

def randomRotate180(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(np.rot90(image))
        mask = np.rot90(np.rot90(mask))
    return image, mask

def randomRotate270(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(np.rot90(np.rot90(image)))
        mask = np.rot90(np.rot90(np.rot90(mask)))
    return image, mask

def randomcrop(image, mask, u=0.5):
    crop_rate = np.random.uniform(0.7,0.9)
    height = np.int32(image.shape[0]*crop_rate)
    width = height
    if np.random.random() < u:
        h, w, c = image.shape
        y = np.random.randint(0, h-height+1)
        x = np.random.randint(0, w-width+1)
        image = image[y:y+height,x:x+width,:]
        image = cv2.resize(image,(h, w), interpolation = cv2.INTER_CUBIC)
        mask = mask[y:y+height,x:x+width]
        mask = cv2.resize(mask,(h, w), interpolation = cv2.INTER_CUBIC)
    return image, mask
