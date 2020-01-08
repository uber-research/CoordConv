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

For more on this project, including a 8-min video explanation, see the [Uber AI Labs blog post](https://eng.uber.com/coordconv).

## CoordConv layer, as a drop-in replacement for convolution
The standalone CoordConv layer, wrapped as a ```tf.layers``` object, can be found in ```CoordConv.py```. Models constructed in ```model_builders.py``` show usage of it.

## Data
To generate *Not-so-Clevr* dataset, which consists of squares randomly positioned on a canvas, and with uniform and quarant splits:
```
python ./data/not_so_clevr_generator.py
```

To generate two-object *Sort-of-Clevr* images, run a modification of the [Sort-of-Clevr source code](https://github.com/kimhc6028/relational-networks/blob/master/sort_of_clevr_generator.py):
```
python ./data/sort_of_clevr_generator.py
```


## Supervised Coordinate Tasks
The ```train.py``` script executes the training of all supervised coordinate tasks as described in the paper. Use ```--arch``` to toggle among different tasks. 

The file ```experiment_logs.sh``` records the entire series of experiments enumerating different hyperparameters for each task, as exactly used to produce results in the paper. Note that we generate random experiment ids for job tracking in the Uber internal cluster, which can be ignored. We also use [resman](https://github.com/yosinski/GitResultsManager) to keep results organized, which is highly recommended!

Examples to run *Supervised Coordinate Classification*:

```
# coordconv version
python train.py --arch coordconv_classification -mb 16 -E 100 -L 0.005 --opt adam --l2 0.001 -mul 1  
# deconv version
python train.py --arch deconv_classification -mb 16 -E 2000 -L 0.01 --opt adam --l2 0.001 -mul 2 -fs 3
```
Use ```--data_h5 data/rectangle_4_uniform.h5``` and ```--data_h5 data/rectangle_4_quadrant.h5``` to observe the performances on two types of splits. 

Examples to run *Supervised Rendering*:

```
# coordconv version
python train.py --arch coordconv_rendering -mb 16 -E 100 -L 0.005 --opt adam --l2 0.001 -mul 1
# deconv version
python train.py --arch deconv_rendering -mb 16 -E 2000 -L 0.01 --opt adam --l2 0.001 -mul 2 -fs 3
```
Use ```--data_h5 data/rectangle_4_uniform.h5``` and ```--data_h5 data/rectangle_4_quadrant.h5``` to observe the performances on two types of splits. 

Examples to run *Supervised Coordinate Regression*:

```
# coordconv version
python train.py --arch conv_regressor -E 100 --lr 0.01 --opt adam --l2 0.00001
# deconv version
python train.py --arch coordconv_regressor -E 100 --lr 0.01 --opt adam --l2 0.00001
```
Use ```--data_h5 data/rectangle_4_uniform.h5``` and ```--data_h5 data/rectangle_4_quadrant.h5``` to observe the performances on two types of splits. 

## Generative Tasks
```
# coordconv GAN
python train_gan.py --arch clevr_coordconv_in_gd -mb 16 -E 50 -L 0.0001 --lr2 .0005 --opt adam --z_dim 256 --snapshot-every 1
# deconv GAN
python train_gan.py --arch clevr_gan -mb 16 -E 50 -L 0.0001 --lr2 .0005 --opt adam --z_dim 256 --snapshot-every 1
```

## TODO
Add RL, and VAE and LSUN GAN models
