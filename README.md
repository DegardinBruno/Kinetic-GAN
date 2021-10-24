# Kinetic-GAN
This repository contains the official PyTorch implementation of the following paper:
> **Generative Adversarial Graph Convolutional Networks for Human Action Synthesis**, Bruno Degardin, João Neves, Vasco Lopes, João Brito, Ehsan Yaghoubi and Hugo Proença, WACV 2022. [[Arxiv Preprint]](https://arxiv.org/abs/2110.11191)

<div align="center">
  <img width="100%" alt="Kinetic-GAN Illustration" src="./utils/simple_demo.gif">
</div>



## Resources

Material related to our paper is available via the following links:

- Paper: https://arxiv.org/abs/2110.11191
- Video: TBA
- Code: https://github.com/DegardinBruno/Kinetic-GAN
- Datasets (ready to use!)
  - NTU RGB+D: [Download](http://socia-lab.di.ubi.pt/~bruno/kinetic-gan/datasets/NTU/ntu.zip) and uncompress it.
  - NTU-120 RGB+D: [Download](http://socia-lab.di.ubi.pt/~bruno/kinetic-gan/datasets/NTU120/ntu-120.zip) and uncompress it.
  - NTU-2D RGB+D: [Download](http://socia-lab.di.ubi.pt/~bruno/kinetic-gan/datasets/NTU2D/ntu-2d.zip) and uncompress it.
  - Human3.6M: [Download](http://socia-lab.di.ubi.pt/~bruno/kinetic-gan/datasets/H36M/h36m.zip) and uncompress it.


## System requirements and Installation

* Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.7+ installation. We recommend pip.
* PyTorch >= 1.7.1
* GPU is not mandatory, but we highly recommend GPU for results reproducibility and speed.

```bash
pip install -r requirements.txt  # use flag --user if permission needed
```

## Model Zoo and Benchmarks

PyTorchVideo provides reference implementation of a large number of video understanding approaches. In this document, we also provide comprehensive benchmarks to evaluate the supported models on different datasets using standard evaluation setup. All the models can be downloaded from the provided links.

### NTU RGB+D

arch     | benchmark | actions | frame length | FID | Config | Model
-------- | --------- | ------- | ------------ | --- | ------ | -----
kinetic-gan-mlp4 | cross-subject | 60 | 64 | 3.618 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)
kinetic-gan-mlp6 | cross-view | 60 | 64 | 4.235 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)

*FID results can differ a bit due to random normal distribution and random noise<br />
** Better action control with MLP-depth 8 (check by yourself with visualization)


### NTU-120 RGB+D

arch     | benchmark | actions | frame length | FID | Config | Model
-------- | --------- | ------- | ------------ | --- | ------ | -----
kinetic-gan-mlp6 | cross-subject | 120 | 64 | 5.967 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)
kinetic-gan-mlp8 | cross-setup | 120 | 64 | 6.751 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)

*FID results can differ a bit due to random normal distribution and random noise<br />
** Better action control with MLP-depth 8 (check by yourself with visualization)


### Human3.6M

arch     | actions | frame length | MMDa | MMDs | Config | Model
-------- | ------- | ------------ | ---- | ---- | ------ | -----
kinetic-gan-mlp6 | 120 | 32 | 0.071 | 0.079 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)
kinetic-gan-mlp8 | 120 | 64 | 0.074 | 0.088 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)
kinetic-gan-mlp8 | 120 | 128 | 0.076 | 0.102 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)
kinetic-gan-mlp8 | 120 | 256 | 0.081 | 0.112 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)
kinetic-gan-mlp8 | 120 | 512 | 0.087 | 0.115 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)
kinetic-gan-mlp8 | 120 | 1024 | 0.092 | 0.121 | [config](http://socia-lab.di.ubi.pt) | [weights](http://socia-lab.di.ubi.pt)

*MMD results can differ a bit due to random normal distribution and random noise<br />
**Additionally, MMD metric is not as "stable" and descriptive as FID, check paper results and visual quality.


## Using pre-trained networks
You can generate your own samples by using a pre-trained Kinetic-GAN with specified [config and weights](https://github.com/DegardinBruno/Kinetic-GAN#model-zoo-and-benchmarks) as folows:

1. Edit or use [generate.py](./generate.py) to specify the dataset where it was trained and arguments.
2. Run the training script with (Check class indexes (label -1) at [NTU RGB+D Datasets](https://rose1.ntu.edu.sg/dataset/actionRecognition/)):
```bash
python generate.py --model model_path  --n_classes number_classes  --label class_index  --gen_qtd how_many_samples  # Check generate.py file
```
3. The experiments (config and samples) are written to a newly created directory `runs/kinetic-gan/exp<id>`.
4. Synthesising is really fast even for huge amounts of samples (GPU recommended but not mandatory).
5. To visualize your samples (`action_ntu.py` for NTU RGB+D and NTU-120 RGB+D and `action_h36m.py` for Human3.6M):
```bash
python visualization/action_ntu.py --path path_samples --labels path_labels --indexes 0 1 2  # Example for Kinetic-GAN trained on NTU or NTU-120
```

## Visualization
You can visualize your samples (`action_ntu.py` for NTU RGB+D and NTU-120 RGB+D and `action_h36m.py` for Human3.6M) by specifying the synthetic samples and labels as follows:
```bash
python visualization/action_ntu.py --path path_samples --labels path_labels --indexes 0 1 2  # Example for Kinetic-GAN trained on NTU or NTU-120, check action_ntu.py file
```

Training will save 10 samples per class at each specified iteration interval. For training with NTU RGB+D, classes are repeated at every 60 samples, run:
```bash
python visualization/action_ntu.py --path path_samples --indexes 26 86 146   # ... Example for `jump up` action.
python visualization/action_ntu.py --path path_samples --indexes 23 83 143   # ... Example for `kicking something` action.
python visualization/action_ntu.py --path path_samples --indexes 58 118 178  # ...Example for `kicking something` action.
```

<div align="center">
  <img width="80%" alt="Blender Actions" src="./utils/actions-cartesian.gif">
</div>


### Blender Visualization
Blender visualization (with mesh) is only applied for a more appealing visualization. For accessing and reproducing our visualization, use specifically our [blender.py](./visualization/blender.py) with [Blender 2.9+ with Python interpreter](https://www.blender.org/download/) (Interpreter already included in Blender). 

<div align="center">
  <img width="100%" alt="Blender Actions" src="./utils/actions-blender.gif">
</div>

Kinetic-GAN generates up to 120 different skeleton actions trained on skeleton-based datasets, which do not have bone rotations specified and dependable by their parent bones, and may cause some poor visualizations with a mesh sometimes (check the initial gif at early iterations).



## Training networks
Datasets are ready to use, after downloading from [resources](https://github.com/DegardinBruno/Kinetic-GAN#resources) you can train your own Kinetic-GAN networks as follows:
1. Edit or use [kinetic-gan.py](./kinetic-gan.py) to specify the dataset and training configuration and arguments.
2. Run the training script with:
```bash
python kinetic-gan.py  --data_path path_train_data.npy  --label_path path_train_labels.pkl  --dataset which_dataset  # check kinetic-gan.py file
```
3. The experiments (files, loss, weights and samples) are written to a newly created directory `runs/kinetic-gan/exp<id>`.
4. For following up the training loss run:
```bash
python visualization/plot_loss.py --batches num_batches_per_epoch --runs kinetic-gan  # check plot_loss.py file
```
5. Training may take up to 48 or 72 hours to complete (using gpu), depending on the configuration and dataset.



