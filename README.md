## Looped Transformers for Length Generalization
This project is for the paper: [Looped Transformers for Length Generalization](https://arxiv.org/abs/2409.15647). Some parts of the codebase are adapted from [in-context-learning](https://github.com/dtsip/in-context-learning) and [transformers](https://github.com/huggingface/transformers).


### Installing the dependencies:
```
git clone https://github.com/UW-Madison-Lee-Lab/looped-tf.git
cd ./looped-tf
conda env create -f environment.yml
conda activate ltf
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```


### Getting started:

First, populate `./src/conf/wandb.yaml` with your wandb info.

To train for specific tasks, please run
```
conda activate ltf
cd ./src
python train.py --conf ./conf/task_name.yaml
```

where `task_name` can be "parity", "copy", "addition", "multi" (for multiplication), "sum_reverse", "dict" (for unique set).
The training log and models would be saved in `./models`.

If you find the code useful please cite:

```
@inproceedings{
fan2025looped,
title={Looped Transformers for Length Generalization},
author={Ying Fan and Yilun Du and Kannan Ramchandran and Kangwook Lee},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=2edigk8yoU}
}
```
