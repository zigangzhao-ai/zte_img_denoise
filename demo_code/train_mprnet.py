import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from warmup_scheduler import GradualWarmupScheduler

from dataloader.data_loader import train_dataloader,test_dataloader
from utils import Adder, Timer, check_lr
from losses import losses
from valid_split import _valid
import numpy as np

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def _train(model, model_running, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # criterion = torch.nn.L1Loss()
    criterion = losses.CharbonnierLoss()
    mse_criterion =  torch.nn.MSELoss(reduce=True, size_average=True)
    l1_criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=args.weight_decay,
                                 )
    ######### Scheduler-warmup+cosine ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch-warmup_epochs+40, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    
    ######### Scheduler-MultiStepLR ###########
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
             
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    epoch = 1
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter()
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = -1
    best_ssim = -1
    Height = 217
    Width = 289

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            # print('----', input_img.size())
            H = input_img.size(3)
            W = input_img.size(4)
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            input_img = input_img.squeeze(dim=1)
            label_img = label_img.squeeze(dim=1)

            for i in range(int(H / Height)):  
                for j in range(int(W/ Width)):
                    input_patch = input_img[:,:,i * Height:i * Height + Height, j * Width:j * Width + Width]
                    label_patch = label_img[:,:,i * Height:i * Height + Height, j * Width:j * Width + Width]
                    #根据Height和Width切成了8*8个patch

                    optimizer.zero_grad()
                    pred_img = model(input_patch)
                    loss_content = np.sum([criterion(torch.clamp(pred_img[j],0,1),label_patch) for j in range(len(pred_img))]) 
                   
                    # label_fft = torch.fft.fft2(label_patch, dim=(-2, -1))
                    # pred_fft = torch.fft.fft2(pred_img[0], dim=(-2, -1)) 
                    # loss_fft = l1_criterion(pred_fft, label_fft)
                    # loss = 0.9 * loss_content + 0.1 * loss_fft
                    loss = loss_content
                    loss.backward()
                    optimizer.step()
                    accumulate(model_running, model)

                    iter_pixel_adder(loss_content.item())
                    # iter_fft_adder(loss_fft.item())
                    epoch_pixel_adder(loss_content.item())
                    # epoch_fft_adder(loss_fft.item())
                    
            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f " % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                # writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                # iter_fft_adder.reset()
    

        if epoch_idx > 100 and  epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pth' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
            torch.save({'model': model_running.state_dict()}, os.path.join(args.model_save_dir, str(epoch_idx).zfill(5)+'running'+'.pth'))

        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f " % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average()))
        # epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % 1 == 0:
            psnr,ssim = _valid(model, args, epoch_idx)
            print('%03d epoch \n Average GOPRO PSNR %.2f dB SSIM %.2f' % (epoch_idx, psnr,ssim))
            writer.add_scalar('PSNR_GOPRO', psnr, epoch_idx)
            writer.add_scalar('SSIM_GOPRO', ssim, epoch_idx)
            if psnr >= best_psnr and ssim >= best_ssim:
                best_psnr = psnr
                best_ssim = ssim
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'best_model.pth'))
    save_name = os.path.join(args.model_save_dir, 'final.pth')
    torch.save({'model': model.state_dict()}, save_name)