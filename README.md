# zte_img_denoise
a contest about image denoising

## Directory structure
```
├── checkpoints
├── data
│   ├── train
│   │   ├── ground_truth
│   │   └── noisy
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
## Running Pose_IDCard
### Experimental environment
* ubuntu16.04 + pytorch1.9.0+ cuda10.2 + python3.8
* You might need 18g memory
### Start
* Run `pip install -r requirement.txt` to install required modules.

### Train
* In the `demo_code` folder, run
```bash
  python main.py
```
### Test
* Place trained model at the checkpoints/UNet/`.

* In the `demo_code` folder, run 
```bash
  python test.py 
```
