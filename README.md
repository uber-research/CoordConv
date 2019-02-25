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
./data/not_so_clevr_generator.py
```

To generate two-object *Sort-of-Clevr* images:
```python
./data/sort_of_clevr_generator.py
```
(A modification of the [Sort-of-Clevr source code](https://github.com/kimhc6028/relational-networks/blob/master/sort_of_clevr_generator.py))

## CoordConv layer
The standalone CoordConv layer, wrapped as a ```tensorflow.python.layers``` object, can be found in ```CoordConv.py``` 
