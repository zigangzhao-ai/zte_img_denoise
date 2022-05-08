import torch
import os
from utils import Adder
import numpy as np
import skimage.metrics
import time
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio
from dataloader.data_loader import test_dataloader
from dataloader.data_process import inv_normalization,write_image,read_image,write_back_dng

def _test(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    for m in model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    black_level = 1024
    white_level = 16383
    Height = 217
    Width = 289
    ep = time.time()

    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(dataloader):
            input_img,name = data
            H = input_img.size(3)
            W = input_img.size(4)
            input_img = input_img.squeeze(dim=1)
            input_img = input_img.to(device)

            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))
            elapsed = time.time() - ep
            adder(elapsed)
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
                    pred = model(j)
                    jj_list.append(pred)
                ii_list.append(jj_list)
            pred = torch.zeros(input_img.size())
            for i_index, h in enumerate(ii_list):
                for j_index, w in enumerate(h):
                    pred[:, :, i_index * Height:i_index * Height + Height, j_index * Width:j_index * Width + Width] = w

            image, height, width = read_image('../data/test/noisy0.dng')
            result_data = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
            result_data = result_data.reshape(-1, height // 2, width // 2, 4)
            result_data = inv_normalization(result_data, black_level, white_level)
            result_write_data = write_image(result_data, height, width)
            print(name[0])
            write_back_dng('../data/test/'+name[0], '../data/result/denoise'+name[0][-5]+'.dng', result_write_data)


            print('%d iter  time: %f' % (idx + 1, elapsed))

        print('==========================================================')
        print("Average time: %f" % adder.average())




