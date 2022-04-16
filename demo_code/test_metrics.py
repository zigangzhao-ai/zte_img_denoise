import os
import numpy as np
import rawpy
import torch
import skimage.metrics
from matplotlib import pyplot as plt
from unetTorch import Unet
import argparse
import pylab

gt_path = '../data/ground_truth/0_gt.dng'
noise_path = '../data/noisy/0_noise.dng'
gt = rawpy.imread(gt_path).raw_image_visible
noise = rawpy.imread(noise_path).raw_image_visible
white_level = 16383
psnr = skimage.metrics.peak_signal_noise_ratio(
    gt.astype(np.float), noise.astype(np.float), data_range=white_level)
ssim = skimage.metrics.structural_similarity(
    gt.astype(np.float), noise.astype(np.float), multichannel=True, data_range=white_level)
print('psnr:', psnr)
print('ssim:', ssim)