## Looped Transformers for Length Generalization
This project is for the paper: [Looped Transformers for Length Generalization](https://arxiv.org/abs/2409.15647). Some parts of the codebase are adapted from [in-context-learning](https://github.com/dtsip/in-context-learning) and [transformers](https://github.com/huggingface/transformers).


### Installing the dependencies:
```
git clone https://github.com/UW-Madison-Lee-Lab/looped-tf.git
cd ./looped-tf
run install.sh
```


### Getting started:

First, populate `./src/conf/wandb.yaml` with your wandb info.

To train for specific tasks, please run
```
cd ./src
python train.py --conf ./conf/task_name.yaml
```

where `task_name` can be "parity", "copy", "addition", "multi" (for multiplication), "sum_reverse", "dict" (for unique set).
The training log and models would be saved in `./models`.
