import torch
from torchvision.transforms import functional as F
from dataloader.data_loader import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
from dataloader.data_process import inv_normalization, write_image, read_image, write_back_dng
import numpy as np
import skimage.metrics


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gopro = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()
    ssim_adder = Adder()
    black_level = 1024
    white_level = 16383
    Height = 217
    Width = 289

    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(gopro):
            input_img, label_img = data
            H = input_img.size(3)
            W = input_img.size(4)
            input_img = input_img.squeeze(dim=1)
            label_img = label_img.squeeze(dim=1)
            input_img = input_img.to(device)
            i_list = []
            for i in range(int(H / Height)):
                j_list = []
                for j in range(int(W / Width)):
                    input_patch = input_img[:, :, i * Height:i * Height + Height, j * Width:j * Width + Width]
                    j_list.append(input_patch)
                i_list.append(j_list)
            ii_list = []
            for i in i_list:
                jj_list = []
                for j in i:
                    pred = model(j)[0] # for u2net
                    # pred = model(j) #for unet
                    jj_list.append(pred)
                ii_list.append(jj_list)
            pred  = torch.zeros(input_img.size())
            for i_index,h in enumerate(ii_list):
                for j_index,w in enumerate(h):
                    pred[:, :, i_index * Height:i_index * Height + Height, j_index * Width:j_index * Width + Width]= w
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))

            image, height, width = read_image('../data/train/ground_truth/1_gt.dng')
            result_data = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
            result_data = result_data.reshape(-1, height // 2, width // 2, 4)
            result_data = inv_normalization(result_data, black_level, white_level)
            result_write_data = write_image(result_data, height, width)
            # write_back_dng('../data/valid/noisy/'+name[0], '../data/valid/sample/epoch'+str(ep).zfill(5)+'_'+name[0], result_write_data)

            gt = label_img.cpu().detach().numpy().transpose(0, 2, 3, 1)
            gt = gt.reshape(-1, height // 2, width // 2, 4)
            gt = inv_normalization(gt, black_level, white_level)
            gt = write_image(gt, height, width)


            """
            obtain psnr and ssim
            """
            psnr = skimage.metrics.peak_signal_noise_ratio(
                gt.astype(np.float), result_write_data.astype(np.float), data_range=white_level)
            ssim = skimage.metrics.structural_similarity(
                gt.astype(np.float), result_write_data.astype(np.float), multichannel=True, data_range=white_level)
            print('psnr:', psnr)
            print('ssim:', ssim)

            psnr_adder(psnr)
            ssim_adder(ssim)
    print('\n')
    model.train()
    return psnr_adder.average(),ssim_adder.average()
