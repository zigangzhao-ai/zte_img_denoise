import os
import torch

from dataloader.data_loader import train_dataloader,test_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid_split import _valid
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler
import torch.optim as optim
from losses.losses import muti_mse_loss_fusion


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)

def _train(model, model_running, args):
    accumulate(model_running, model, 0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mse_criterion = torch.nn.MSELoss()
    l1_criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=args.weight_decay,
                                 )

    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)

    ######### Scheduler-warmup+cosine ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch-warmup_epochs+40, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    # scheduler.step()
    ######### Scheduler-MultiStepLR ###########
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)

    epoch = 1
    if args.pretrained:
        print("====", args.pretrained)
        state_dict = torch.load(args.pretrained)
        model.load_state_dict(state_dict['model'])
        
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
                    # pred_patch = model(input_patch)
                    # loss_content = mse_criterion(pred_patch, label_patch)
                    d0, d1, d2, d3, d4, d5, d6 = model(input_patch)
                    loss2, loss_content = muti_mse_loss_fusion(d0, d1, d2, d3, d4, d5, d6, label_patch)
                    
                    label_fft = torch.fft.fft2(label_patch, dim=(-2, -1))
                    pred_fft = torch.fft.fft2(d0, dim=(-2, -1)) 
                    loss_fft = l1_criterion(pred_fft, label_fft)
                    loss = 0.9 * loss_content + 0.1 * loss_fft
                    # loss = loss_content

                    loss.backward()
                    optimizer.step()
                    accumulate(model_running, model)

                    iter_pixel_adder(loss_content.item())
                    iter_fft_adder(loss_fft.item())

                    epoch_pixel_adder(loss_content.item())
                    epoch_fft_adder(loss_fft.item())
                 
            # for i in range(int(H / Height)):
            #     j_list = []
            #     for j in range(int(W / Width)):
            #         input_patch = input_img[:, :, i * Height:i * Height + Height, j * Width:j * Width + Width]
            #         # 根据Height和Width切成了8*8个patch
            #         pred_patch = model(input_patch)
            #         j_list.append(pred_patch)
            #     i_list.append(j_list)
            # #由于切成patch再合并会有边界噪声，所以合并再计算了一个loss，尽可能抑制边界
            # pred = torch.zeros(input_img.size()).cuda()
            # for i_index, h in enumerate(i_list):
            #     for j_index, w in enumerate(h):
            #         pred[:, :, i_index * Height:i_index * Height + Height, j_index * Width:j_index * Width + Width] = w
            # optimizer.zero_grad()
            # l1_loss = criterion(pred, label_img)
            # loss_content = l1_loss*64
            # label_fft = torch.rfft(label_img, signal_ndim=2, normalized=False, onesided=False)
            # pred_fft = torch.rfft(pred, signal_ndim=2, normalized=False, onesided=False)
            # f = criterion(pred_fft, label_fft)*64
            # loss_fft = f
            # loss = loss_content + 0.1 * loss_fft
            # loss.backward()
            # optimizer.step()
            # lr = check_lr(optimizer)
            # print(" Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.9f Loss fft: %7.9f" % ( epoch_idx, iter_idx + 1, max_iter, lr, loss_content.item(),
            #     loss_fft.item()))
            
            if (iter_idx + 1) % args.print_freq == 0:
                lr = check_lr(optimizer)
                print("Time: %7.4f Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f Loss fft: %7.4f" % (
                    iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter, lr, iter_pixel_adder.average(),
                    iter_fft_adder.average()))
                writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)
                iter_timer.tic()
                iter_pixel_adder.reset()
                iter_fft_adder.reset()
    
        if epoch_idx > 100 and  epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pth' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)
            #if need resume, use this method to save model 
            # torch.save({'model': model.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'scheduler': scheduler.state_dict(),
            #             'epoch': epoch_idx}, save_name)
            torch.save({'model': model_running.state_dict()}, os.path.join(args.model_save_dir, str(epoch_idx).zfill(5)+'running'+'.pth'))
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
        epoch_pixel_adder.reset()
        scheduler.step()
        if epoch_idx % 1 == 0:
            psnr,ssim = _valid(model, args, epoch_idx)
            # torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, str(epoch_idx).zfill(5)+'.pth'))
            print('%03d epoch \n Average GOPRO PSNR %.2f dB SSIM %.2f' % (epoch_idx, psnr,ssim))
            writer.add_scalar('PSNR_GOPRO', psnr, epoch_idx)
            writer.add_scalar('SSIM_GOPRO', ssim, epoch_idx)

            # psnr, ssim = _valid(model_running, args, epoch_idx)
            # torch.save({'model': model_running.state_dict()}, os.path.join(args.model_save_dir, str(epoch_idx).zfill(5)+'running'+'.pth'))
            # print('%03d epoch \n Average  Running GOPRO PSNR %.2f dB SSIM %.2f' % (epoch_idx, psnr, ssim))
            if psnr >= best_psnr and ssim >= best_ssim:
                best_psnr = psnr
                best_ssim = ssim
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'best_model.pth'))
    save_name = os.path.join(args.model_save_dir, 'final.pth')
    torch.save({'model': model.state_dict()}, save_name)