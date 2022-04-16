import torch
from torchvision.transforms import functional as F
from dataloader.data_loader import test_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
from dataloader.data_process import inv_normalization, write_image, read_image, write_back_dng
import numpy as np
import skimage.metrics
import time

def _test(model, args):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    adder = Adder()
    model.eval()
    black_level = 1024
    white_level = 16383
    ep = time.time()

    with torch.no_grad():
        print('Start GoPro Evaluation')
        for idx, data in enumerate(dataloader):
            input_img,name = data
            input_img = input_img.squeeze(dim=1)
            input_img = input_img.to(device)
            if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
                os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))
            elapsed = time.time() - ep
            adder(elapsed)
            pred = model(input_img)
            image, height, width = read_image('../data/test/noisy0.dng')
            result_data = pred.cpu().detach().numpy().transpose(0, 2, 3, 1)
            result_data = result_data.reshape(-1, height // 2, width // 2, 4)
            result_data = inv_normalization(result_data, black_level, white_level)
            result_write_data = write_image(result_data, height, width)
            write_back_dng('../data/test/'+name[0], '../data/result/predict'+name[0], result_write_data)

            print('%d iter  time: %f' % (idx + 1, elapsed))
        print('==========================================================')
        print("Average time: %f" % adder.average())




