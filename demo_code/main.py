import os
import torch
import argparse
from torch.backends import cudnn
from net.Unet import Unet
from net.UNetplusplus import UNetplusplus
from net.Unet_ACNet import Unet_ACNet
from train_unet import _train
from test_unet import _test

def main(args):
    # CUDNN
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    model = Unet()
    # print(model)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    elif args.mode == 'test':
        _test(model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--model_name', default='UNet', choices=['UNet','MIMO-UNet', 'MIMO-UNetPlus'], type=str)
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    # Train
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4) # default=1e-4
    parser.add_argument('--weight_decay', type=float, default=1e-8) ## default=0, 5e-2
    parser.add_argument('--num_epoch', type=int, default=400)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x+2) * 40 for x in range(400//50)])

    # Test
    parser.add_argument('--test_model', type=str, default='../checkpoints/UNet/Best.pth')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])

    args = parser.parse_args()
    args.model_save_dir = os.path.join('../checkpoints/', args.model_name)
    args.result_dir = os.path.join('../results/', args.model_name, 'result_image/')
    print(args)
    main(args)
