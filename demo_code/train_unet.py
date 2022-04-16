import os
import torch

from dataloader.data_loader import train_dataloader,test_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F


def _train(model, args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # criterion = torch.nn.L1Loss()
    mse_criterion =  torch.nn.MSELoss(reduce=True, size_average=True)
    l1_criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay,
                                 betas=(0.9, 0.999))
                
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
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

    for epoch_idx in range(epoch, args.num_epoch + 1):

        epoch_timer.tic()
        iter_timer.tic()
        for iter_idx, batch_data in enumerate(dataloader):
            input_img, label_img = batch_data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            input_img = input_img.squeeze(dim = 1)
            label_img = label_img.squeeze(dim = 1)

            optimizer.zero_grad()
            pred_img = model(input_img)         

            loss_content = mse_criterion(pred_img, label_img)
            label_fft = torch.fft.fft2(label_img, dim=(-2, -1))
            pred_fft = torch.fft.fft2(pred_img, dim=(-2, -1))
            loss_fft = l1_criterion(pred_fft, label_fft)

            loss = 0.8 * loss_content + 0.2 * loss_fft
            loss.backward()
            optimizer.step()

            iter_pixel_adder(loss_content.item())
            iter_fft_adder(loss_fft.item())

            epoch_pixel_adder(loss_content.item())
            epoch_fft_adder(loss_fft.item())

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
        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average(), epoch_fft_adder.average()))
        epoch_fft_adder.reset()
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