# Kinetic-GAN
This repository contains the official PyTorch implementation of the following paper:
> **Generative Adversarial Graph Convolutional Networks for Human Action Synthesis**, Bruno Degardin, João Neves, Vasco Lopes, João Brito, Ehsan Yaghoubi and Hugo Proença, WACV 2022. [[Arxiv Preprint]](https://arxiv.org/abs/2110.11191)

## Resources

Material related to our paper is available via the following links:

- Paper: https://arxiv.org/abs/2110.11191
- Video: TBA
- Code: https://github.com/DegardinBruno/Kinetic-GAN
- Datasets
  - NTU RGB+D: https://github.com/NVlabs/ffhq-dataset
  - NTU-120 RGB+D: https://github.com/NVlabs/ffhq-dataset
  - NTU-2D RGB+D: https://github.com/NVlabs/ffhq-dataset
  - Human3.6M: https://github.com/NVlabs/ffhq-dataset


## Model Zoo and Benchmarks

PyTorchVideo provides reference implementation of a large number of video understanding approaches. In this document, we also provide comprehensive benchmarks to evaluate the supported models on different datasets using standard evaluation setup. All the models can be downloaded from the provided links.

### NTU RGB+D

arch     | depth | frame length x sample rate | top 1 | top 5 | Flops (G) x views | Params (M) | Model
-------- | ----- | -------------------------- | ----- | ----- | ----------------- | ---------- | -------------------------------------------------------------------------------------
C2D      | R50   | 8x8                        | 71.46 | 89.68 | 25.89 x 3 x 10    | 24.33      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/C2D\_8x8\_R50.pyth)
I3D      | R50   |8x8                        | 73.27 | 90.70 | 37.53 x 3 x 10    | 28.04      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/I3D\_8x8\_R50.pyth)
Slow     | R50   |4x16                       | 72.40 | 90.18 | 27.55 x 3 x 10    | 32.45      | [link](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/SLOW\_4x16\_R50.pyth)

### Download NTU RGB+D X-Subject Dataset
FileZilla -> IP: 10.0.4.137 -> video -> needed_DEGARDIN -> DATASETS -> NTU-RGBD -> xsub -> train_data.npy and train_label.pkl


### Kinetic-GAN
Kinetic-GAN with NTU RGB+D xsub dataset as input

#### Train
```
python kinetic-gan.py --data_path path_train_data --label_path path_train_labels
```

---

### Visualization
Visualization of synthetic samples. Check [NTU-RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset labels (index=label-1).
Generator synthesizes 10 samples from the 60 classes. Classes are repeated at every 60 samples, check "jump up" example below.

```
python visualization/synthetic.py --path path_samples --index_sample 26 86 146  # Multiple samples indexes (Max 3) 
```

Check current loss.
```
python visualization/plot_loss.py  # Check inside settings
```
