# Graph_GAN
Code for the paper "Generative Adversarial Graph Convolutional Networks to Synthesize Human Actions ", WACV 2022


### cGAN - Skeleton Input
Conditional-GAN adapted with NTU-RGB+D as input

#### Train
```
python cgan_graph.py --data_path path_train_data --label_path path_train_labels
```


### StarGAN - Skeleton Input
StarGAN adapted with NTU-RGB+D as input

#### Train
```
python stargan_graph.py --train_path path_train_data --train_label_path path_train_labels --val_path path_val_data --val_label_path path_val_labels
```


### Graph-Autoencoder - Skeleton Input
Graph-Autoencoder adapted with NTU-RGB+D as input

#### Train
```
cd gans/graph-ae;
python graph-ae.py 
```