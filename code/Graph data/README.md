# DAST: Domain-Agnostic Self-Training for Semi-Supervised Learning
This is an PyTorch implementation of DAST for graphs.

![image-20230125133836939](img/image-20230125133836939.png)



## Requirements

We recommend using `anaconda` or `miniconda` for python. Our code has been tested with `python=3.8` on linux.

Make sure the following requirements are met

* torch>=1.8.1
* torchvision>=0.9.1

- torch_cluster
- torch_scatter
- torch_sparse
- torch_spline_conv
- [torch-geometric](https://github.com/rusty1s/pytorch_geometric) >= 1.6.0
- [ogb](https://github.com/snap-stanford/ogb) == 1.2.4



## Training & Evaluation

In each of our experiments, we use a single Nvidia GeForce RTX 2080Ti GPU.

### Train

Train the model  of MUTAG dataset:

```
python main.py --dataset MUTAG --split_ratio 0.05  --epochs 100 --lr 0.002 --gamma_joao 0.1  --suffix 0
```

### Arguments

* `--dataset` : Dataset from Tudataset
* `--split_ratio` : Label ratio
* `--epochs` : Training epochs
* `--lr` : Learning rate
* `--gamma_joao` : We use the defualt number following JOAO
* `--suffix` : We use the defualt number following JOAO

#### <span style="color:Tomato">Most of the hyperparameters are hardcoded in train.py file.</span>

### Evaluation

We choose the best model by evaluating the model on validation dataset. The accuracy of the best model on test datasets is printed after training is completed.





