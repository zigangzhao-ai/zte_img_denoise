## 2022 中兴捧月大赛（多媒体赛道）-图像去噪 Baseline

## 环境创建
1.python环境创建

    conda create --name env_demo python=3.6

2.安装软件包

    pip install -r requirements.txt

3.运行baseline

    python ./demo_code/testTorch.py --model_path=./models/th_model.pth \ 
                                    --black_level=1024 \
                                    --white_level=16383 \
                                    --input_path=./data/noise/demo_noise.dng \
                                    --output_path=./data/result/demo_torch_res.dng \
                                    --ground_path=./data/gt/demo.dng 
    
