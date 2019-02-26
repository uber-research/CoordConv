# CoordConv

This repository contains source code necessary to reproduce the results presented in the paper [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247) (NeurIPS 2018):

```
@inproceedings{liu2018coordconv,
  title={An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution},
  author={Liu, Rosanne and Lehman, Joel and Molino, Piero and Petroski Such, Felipe and Frank, Eric and Sergeev, Alex and Yosinski, Jason},
  booktitle={Advances in Neural Information Processing Systems},
  year={2018}
}
```

For more on this project, see the [Uber AI Labs Blog post](https://eng.uber.com/coordconv).

## Data
To generate *Not-so-Clevr* dataset, which consists of squares randomly positioned on a canvas, and with uniform and quarant splits:
```python
python ./data/not_so_clevr_generator.py
```

To generate two-object *Sort-of-Clevr* images, run a modification of the [Sort-of-Clevr source code](https://github.com/kimhc6028/relational-networks/blob/master/sort_of_clevr_generator.py):
```python
python ./data/sort_of_clevr_generator.py
```

## CoordConv layer
The standalone CoordConv layer, wrapped as a ```tf.layers``` object, can be found in ```CoordConv.py``` 

## Supervised Coordinate Tasks
The ```train.py``` script executes the training of all supervised coordinate tasks as described in the paper. Use ```--arch``` to toggle among different tasks.

To run *Supervised Coordinate Classification*:

```python
# coordconv version
python train.py --arch coordconv_classification -mb 16 -E 100 -L 0.005 --opt adam --l2 0.001 -mul 1  
# deconv version
python train.py --arch deconv_classification -mb 16 -E 2000 -L 0.01 --opt adam --l2 0.001 -mul 2 -fs 3
```
Use ```--data_h5 data/rectangle_4_uniform.h5``` and ```--data_h5 data/rectangle_4_quadrant.h5``` to observe the performances on two types of splits. 

To run *Supervised Rendering*:

```python
# coordconv version
python train.py --arch coordconv_rendering -mb 16 -E 100 -L 0.005 --opt adam --l2 0.001 -mul 1
# deconv version
python train.py --arch deconv_rendering -mb 16 -E 2000 -L 0.01 --opt adam --l2 0.001 -mul 2 -fs 3
```
Use ```--data_h5 data/rectangle_4_uniform.h5``` and ```--data_h5 data/rectangle_4_quadrant.h5``` to observe the performances on two types of splits. 

To run *Supervised Coordinate Regression*:

```python
# coordconv version
python train.py --arch conv_regressor -E 100 --lr 0.01 --opt adam --l2 0.00001
# deconv version
python train.py --arch coordconv_regressor -E 100 --lr 0.01 --opt adam --l2 0.00001
```
Use ```--data_h5 data/rectangle_4_uniform.h5``` and ```--data_h5 data/rectangle_4_quadrant.h5``` to observe the performances on two types of splits. 

## Generative Tasks
```python
python train_gan.py --arch simple_gan -mb 16 -E 50 -L 0.0001 --lr2 .0005 --opt adam --z_dim 25 --snapshot-every 1
python train_gan.py --arch clevr_gan -mb 16 -E 50 -L 0.0001 --lr2 .0005 --opt adam --z_dim 25 --snapshot-every 1
```

## TODO
Add RL, and VAE and LSUN GAN models
