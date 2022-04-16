import os
import numpy as np
import rawpy
import torch
import skimage.metrics
from matplotlib import pyplot as plt
from unetTorch import Unet
import argparse
import pylab
from dataloader.data_process import read_image, normalization, inv_normalization, write_image, write_back_dng


def denoise_raw(input_path, output_path, ground_path, model_path, black_level, white_level):
    """
    Example: obtain ground truth
    """
    gt = rawpy.imread(ground_path).raw_image_visible 

    """
    pre-process
    """
    raw_data_expand_c, height, width = read_image(input_path)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
    raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        raw_data_expand_c_normal.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float().cuda()
    net = Unet()
    net = net.cuda()
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    net.eval()

    """
    inference
    """
    result_data = net(raw_data_expand_c_normal)

    """
    post-process
    """
    result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
    result_data = inv_normalization(result_data, black_level, white_level)
    result_write_data = write_image(result_data, height, width)
    write_back_dng(input_path, output_path, result_write_data)

    """
    obtain psnr and ssim
    """
    psnr = skimage.metrics.peak_signal_noise_ratio(
        gt.astype(np.float), result_write_data.astype(np.float), data_range=white_level)
    ssim = skimage.metrics.structural_similarity(
        gt.astype(np.float), result_write_data.astype(np.float), multichannel=True, data_range=white_level)
    print('psnr:', psnr)
    print('ssim:', ssim)

    """
    Example: this demo_code shows your input or gt or result image
    """
    f0 = rawpy.imread(ground_path)
    f1 = rawpy.imread(input_path)
    f2 = rawpy.imread(output_path)
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(f0.postprocess(use_camera_wb=True))
    axarr[1].imshow(f1.postprocess(use_camera_wb=True))
    axarr[2].imshow(f2.postprocess(use_camera_wb=True))
    axarr[0].set_title('gt')
    axarr[1].set_title('noisy')
    axarr[2].set_title('de-noise')

def denoise_raw_pure(input_path, output_path,  model_path, black_level, white_level):

    """
    pre-process
    """
    raw_data_expand_c, height, width = read_image(input_path)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
    raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        raw_data_expand_c_normal.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()
    net = Unet()
    if model_path is not None:
        # net.load_state_dict(torch.load(model_path))
        state_dict = torch.load(model_path)
        net.load_state_dict(state_dict['model'],strict=False)

    net.eval()

    """
    inference
    """
    result_data = net(raw_data_expand_c_normal)

    """
    post-process
    """
    result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
    print('----', result_data.shape) #(1, 1736, 2312, 4)
    result_data = inv_normalization(result_data, black_level, white_level)
    # result_data = result_data.reshape(-1, height // 2, width // 2, 4)
    # result_data = inv_normalization(result_data, black_level, white_level)
    result_write_data = write_image(result_data, height, width)
    write_back_dng(input_path, output_path, result_write_data)


def main(args):
    model_path = args.model_path
    black_level = args.black_level
    white_level = args.white_level
    input_path = args.input_path
    output_path = args.output_path
    ground_path = args.ground_path
    denoise_raw(input_path, output_path, ground_path, model_path, black_level, white_level)


if __name__ == '__main__':

    import glob
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="../models/th_model.pth")
    parser.add_argument('--black_level', type=int, default=1024)
    parser.add_argument('--white_level', type=int, default=16383)
    parser.add_argument('--input_path', type=str, default="../data/noisy/0_noise.dng")
    parser.add_argument('--output_path', type=str, default="../data/result/demo_torch_res.dng")
    parser.add_argument('--ground_path', type=str, default="../data/ground truth/0_gt.dng")

    args = parser.parse_args()
    # main(args)

    output_dir = "0415/out1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imgnames = glob.glob("/testset/*.dng")
    for img_name in imgnames:
        print(img_name)
        input_path = img_name
        tag = os.path.basename(img_name)[-5] #5
        output_path = os.path.join(output_dir, 'denoise'+ tag +'.dng')
        denoise_raw_pure(input_path, output_path, args.model_path, args.black_level, args.white_level)
    

