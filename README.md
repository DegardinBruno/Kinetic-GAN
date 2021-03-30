# Graph_GAN
Code for the paper "Generative Adversarial Graph Convolutional Networks to Synthesize Human Actions"


---
### dc-GAN
Deep Convolutional GAN adapted with NTU-RGB+D as input

#### Train
```
python dcgan-graph.py --data_path path_train_data --label_path path_train_labels
```

#
### cGAN
Conditional-GAN adapted with NTU-RGB+D as input

#### Train
```
python cgan_graph.py --data_path path_train_data --label_path path_train_labels
```

---

### Visualization
Visualization of synthetic samples. Check [NTU-RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset labels (index=label-1).
Generator synthesizes 60 samples from the 60 classes. Classes are repeated at every 60 samples, check "jump up" example below.

```
python visualization/synthetic.py --path path_samples --index_sample 26 86 146  # Multiple samples indexes (Max 3) 
```

Check current loss.
```
python visualization/plot_loss.py  # Check inside settings
```
