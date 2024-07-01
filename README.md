# Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification

> Official PyTorch implementation of ["Exploring Fine-Grained Representation and Recomposition for Cloth-Changing Person Re-Identification"](https://arxiv.org/abs/2308.10692). ([TIFS 2024](https://ieeexplore.ieee.org/document/10557733))
>
> Qizao Wang, Xuelin Qian, Bin Li, Xiangyang Xue, Yanwei Fu
>
> Fudan University, Northwestern Polytechnical University



## Getting Started

### Environment

- Python == 3.8
- PyTorch == 1.12.1
- faiss-gpu == 1.7.2

### Prepare Data

Please download cloth-changing person re-identification datasets and place them in any path `DATASET_ROOT`:

    DATASET_ROOT
    	└─ LTCC-reID or Celeb-reID or PRCC or DeepChange or LaST
    		├── train
    		├── query
    		├── gallery


### Training

```sh
# LTCC
python main.py --gpu_devices 0 --dataset ltcc --dataset_root DATASET_ROOT --dataset_filename LTCC-reID --save_dir SAVE_DIR --save_checkpoint

# Celeb-reID
python main.py --gpu_devices 0 --dataset celeb --dataset_root DATASET_ROOT --dataset_filename Celeb-reID --num_instances 4 --save_dir SAVE_DIR --save_checkpoint

# PRCC
python main.py --gpu_devices 0,1 --dataset prcc --dataset_root DATASET_ROOT --dataset_filename PRCC --max_epoch 30 --save_dir SAVE_DIR --save_checkpoint

# DeepChange
python main.py --gpu_devices 0,1 --dataset deepchange --dataset_root DATASET_ROOT --dataset_filename DeepChange --train_batch 64 --fg_start_epoch 45 --save_dir SAVE_DIR --save_checkpoint

# LaST
python main.py --gpu_devices 0,1 --dataset last --dataset_root DATASET_ROOT --dataset_filename LaST --train_batch 64 --num_instances 4 --fg_start_epoch 45 --save_dir SAVE_DIR --save_checkpoint
```

`--dataset_root` : replace `DATASET_ROOT` with your dataset root path

`--save_dir`: replace `SAVE_DIR` with the path to save log file and checkpoints

It is worth mentioning that adjusting the scanning radius of DBSCAN (by setting `--eps`) can explore fine-grained information of different granularities. 
Increasing the value of `eps` on difficult datasets (e.g., LTCC and DeepChange) may reduce noise and bring slightly better performance, while `eps=0.4` mostly works well.


### Evaluation

```sh
python main.py --gpu_devices 0 --dataset DATASET --dataset_root DATASET_ROOT --dataset_filename DATASET_FILENAME --resume RESUME_PATH --save_dir SAVE_DIR --evaluate
```

`--dataset`: replace `DATASET` with the dataset name

`--dataset_filename`: replace `DATASET_FILENAME` with the folder name of the dataset

`--resume`: replace `RESUME_PATH` with the path of the saved checkpoint

The above three arguments are set corresponding to Training.


### Results

- **Celeb-reID**

| Backbone  | Rank-1 | Rank-5 | mAP  |
| :-------: |:------:|:------:|:----:|
| ResNet-50 |  64.0  |  78.8  | 18.2 |

- **LTCC**

| Backbone  |    Setting     | Rank-1 | mAP  |
| :-------: | :------------: |:------:|:----:|
| ResNet-50 | Cloth-Changing |  44.6  | 19.1 |
| ResNet-50 |    Standard    |  75.9  | 39.9 |

- **PRCC**

| Backbone  |    Setting     | Rank-1 | mAP  |
| :-------: | :------------: |:------:|:----:|
| ResNet-50 | Cloth-Changing |  65.0  | 63.1 |
| ResNet-50 |    Standard    |  100   | 99.5 |

- **DeepChange**

| Backbone  | Rank-1 | mAP  |
| :-------: |:------:|:----:|
| ResNet-50 |  57.9  | 20.0 |

- **LaST**

| Backbone  | Rank-1 | mAP  |
| :-------: |:------:|:----:|
| ResNet-50 |  75.0  | 32.2 |

You can achieve similar results with the released code.

## Citation

Please cite the following paper in your publications if it helps your research:

```
@article{wang2024exploring,
  title={Exploring fine-grained representation and recomposition for cloth-changing person re-identification},
  author={Wang, Qizao and Qian, Xuelin and Li, Bin and Xue, Xiangyang and Fu, Yanwei},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={19},
  pages={6280-6292},
  year={2024},
  publisher={IEEE}
}
```


## Contact

Any questions or discussions are welcome!

Qizao Wang (<qzwang22@m.fudan.edu.cn>)