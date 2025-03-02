# CBLane

# Dateset

* [TuSimple](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download)
* [Curvelanes](https://www.kaggle.com/datasets/bnyadmohammed/curvelanes)
* [CULane](https://xingangpan.github.io/projects/CULane.html)

# Install
Please see [INSTALL.md](./INSTALL.md)

# Get started
Please modify the `data_root` in any configs you would like to run. We will use `configs/culane_res18.py` as an example.

To train the model, you can run:
```
python train.py configs/culane_res18.py --log_path /path/to/your/work/dir
```
or
```
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/culane_res18.py --log_path /path/to/your/work/dir
```
It should be noted that if you use different number of GPUs, the learning rate should be adjusted accordingly. The configs' learning rates correspond to 8-GPU training on CULane and CurveLanes datasets. **If you want to train on CULane or CurveLanes with single GPU, please decrease the learning rate by a factor of 1/8.** On the Tusimple, the learning rate corresponds to single GPU training.
# Trained models
For evaluation, run
```Shell
mkdir tmp

python test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
```

Same as training, multi-gpu evaluation is also supported.
```Shell
mkdir tmp

python -m torch.distributed.launch --nproc_per_node=8 test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
```

# Visualization
We provide a script to visualize the detection results. Run the following commands to visualize on the testing set of CULane.
```
python demo.py configs/culane_res18.py --test_model /path/to/your/culane_res18.pth
```
# Attention

* The hyperparameters need to be set manually.
* Various paths in the code need to be reconfigured by yourself.
* The setting of the dataset needs to be done by yourself.

# References
* [YaoleiQi/DSCNet](https://github.com/YaoleiQi/DSCNet)
* [houqb/CoordAttention](https://github.com/houqb/CoordAttention)
* [cfzd/Ultra-Fast-Lane-Detection-v2](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)