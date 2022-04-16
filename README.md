# zte_img_denoise
a contest about image denoising

# dir
```
├── checkpoints
│   └── UNet
│       ├── Best.pth
│       └── model.pth
├── data
│   ├── train
│   │   ├── ground_truth
│   │   │   ├── 0_gt.dng
│   │   │   └── 99_gt.dng
│   │   └── noisy
│   │       ├── 0_noise.dng
│   │       └── 99_noise.dng
│   └── valid
│       ├── ground_truth
│       └── noisy
├── demo_code
│   ├── dataloader
│   │   ├── data_augment.py
│   │   ├── data_loader.py
│   │   └── data_process.py
│   ├── main.py
│   ├── test.py
│   ├── test_metrics.py
│   ├── test_unet.py
│   ├── train_unet.py
│   ├── unetTorch.py
│   ├── utils.py
│   └── valid.py
├── readme.md
└── requirements.txt

```

