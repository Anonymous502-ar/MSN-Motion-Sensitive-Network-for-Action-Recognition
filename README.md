## Acknowledgements
The code based on the [TDN](https://github.com/MCG-NJU/TDN), We especially thank the contributors of this excellent work.

# MSN-Motion-Sensitive-Network-for-Action-Recognition
* [Prerequisites](#prerequisites)
* [Data Preparation](#data-preparation)
* [Model Zoo](#model-zoo)
* [Testing](#testing)  
* [Training](#training)  

## Prerequisites
The code is built with following libraries:

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) **1.4** or higher
- [Torchvision](https://github.com/pytorch/vision)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [scikit-learn](https://scikit-learn.org/stable/)
- [ffmpeg](https://www.ffmpeg.org/)  
- [decord](https://github.com/dmlc/decord) 

## Data Preparation
We have successfully trained TDN on [Kinetics400](https://deepmind.com/research/open-source/kinetics), [UCF101](https://www.crcv.ucf.edu/data/UCF101.php), [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/), [Something-Something-V1](https://20bn.com/datasets/something-something/v1) and [V2](https://20bn.com/datasets/something-something/v2) with this codebase.  
- The processing of **Something-Something-V1 & V2** can be summarized into 3 steps:
    1. Extract frames from videos(you can use ffmpeg to get frames from video)      
    2. Generate annotations needed for dataloader ("<path_to_frames> <frames_num> <video_class>" in annotations) The annotation usually includes train.txt and val.txt. The format of *.txt file is like:
        ```
        dataset_root/frames/video_1 num_frames label_1
        dataset_root/frames/video_2 num_frames label_2
        dataset_root/frames/video_3 num_frames label_3
        ...
        dataset_root/frames/video_N num_frames label_N
        ```
    3. Add the information to `ops/dataset_configs.py`.

- The processing of **Kinetics400** can be summarized into 3 steps:
    1. We preprocess our data by resizing the short edge of video to 320px. You can refer to [MMAction2 Data Benchmark](https://github.com/open-mmlab/mmaction2) for [TSN](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/tsn#kinetics-400-data-benchmark-8-gpus-resnet50-imagenet-pretrain-3-segments) and [SlowOnly](https://github.com/open-mmlab/mmaction2/tree/master/configs/recognition/slowonly#kinetics-400-data-benchmark).
    2. Generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations) The annotation usually includes train.txt and val.txt. The format of *.txt file is like:
        ```
        dataset_root/video_1.mp4  label_1
        dataset_root/video_2.mp4  label_2
        dataset_root/video_3.mp4  label_3
        ...
        dataset_root/video_N.mp4  label_N
        ```
    3. Add the information to `ops/dataset_configs.py`.

    **Note**:
    We use [decord](https://github.com/dmlc/decord) to decode the Kinetics videos **on the fly**.

## Model Zoo
#### Something-Something-V1

Model  | Frames x Crops x Clips  | Top-1  | Top-5 | checkpoint
:--: | :--: | :--: | :--:| :--:
MSN  | 8x1x1 | 53.0%  | 81.5% | we will release later
MSN  | 16x1x1 | 54.1%  | 82.3% | we will release later

#### Something-Something-V2

Model  | Frames x Crops x Clips | Top-1  | Top-5 | checkpoint
:--: | :--: | :--: | :--:| :--:
MSN  | 8x1x1   | 63.9%   | 89.2%  | we will release later
MSN  | 16x1x1  | 65.5%   | 89.9%  | we will release later

#### Kinetics400
Model  | Frames x Crops x Clips   | Top-1 (30 view)  | Top-5 (30 view)  | checkpoint
:--: | :--: | :--: | :--:| :--:
MSN    | 8x3x10  | 77.1%  | 93.6%  | we will release later

## Testing
- For center crop single clip, the processing of testing can be summarized into 2 steps:
    1. Run the following testing scripts:
        ```
        CUDA_VISIBLE_DEVICES=0 python3 test_models_center_crop.py something \
        --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8  \
        --test_crops=1 --batch_size=16  --gpus 0 --output_dir <your_pkl_path> -j 4 --clip_index=0
        ```
    2. Run the following scripts to get result from the raw score:
        ```
        python3 pkl_to_results.py --num_clips 1 --test_crops 1 --output_dir <your_pkl_path>  
        ```
- For 3 crops, 10 clips, the processing of testing can be summarized into 2 steps: 
    1. Run the following testing scripts for 10 times(clip_index from 0 to 9):
        ``` 
        CUDA_VISIBLE_DEVICES=0 python3 test_models_three_crops.py  kinetics \
        --archs='resnet50' --weights <your_checkpoint_path>  --test_segments=8 \
        --test_crops=3 --batch_size=16 --full_res --gpus 0 --output_dir <your_pkl_path>  \
        -j 4 --clip_index <your_clip_index>
        ```
    2. Run the following scripts to ensemble the raw score of the 30 views:
        ```
        python pkl_to_results.py --num_clips 10 --test_crops 3 --output_dir <your_pkl_path> 
        ```
## Training
This implementation supports multi-gpu, `DistributedDataParallel` training, which is faster and simpler. 
- For example, to train MSN on Something-Something-V1 with 2 gpus, you can run:
    ```
    python -m torch.distributed.launch --master_port 12347 --nproc_per_node=2 \
                main.py  something  RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.01 \
                --lr_scheduler step --lr_steps  30 45 55 --epochs 60 --batch-size 8 \
                --wd 5e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb 
    ```
- For example, to train MSN on Kinetics400 with 2 gpus, you can run:
    ```
    python -m torch.distributed.launch --master_port 12347 --nproc_per_node=2 \
            main.py  kinetics RGB --arch resnet50 --num_segments 8 --gd 20 --lr 0.02 \
            --lr_scheduler step  --lr_steps 50 75 90 --epochs 100 --batch-size 8 \
            --wd 1e-4 --dropout 0.5 --consensus_type=avg --eval-freq=1 -j 4 --npb 
    ```


## License
This repository is released under the Apache-2.0. license as found in the [LICENSE](https://github.com/Anonymous502-ar/MSN-Motion-Sensitive-Network-for-Action-Recognition/LICENSE) file.
