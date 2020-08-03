# BBVI and IAF for Variance Inference
This package contains the code that was used to create the additional experiments for our "VIPS"-paper. 

It contains an implementation of the following methods for variational inference:
* Inverse Autoregressive Flows
* Black-Box Variational Inference for training GMMs

It contains an implementation of the following experiments:
* German Credit
* Breast Cancer
* Planer N-Link
* Target GMM

## Installation
The packages requires Python 3 and Tensorflow 2. 
It can be installed by (creating a virtual environment and) running 
```bash
pip install -r requirements.txt
```

### Getting Started
The experiments can be started by running the scripts found in "experiments/<method>/", e.g.  [experiments/BBVI/logist_regression.py](experiments/BBVI/logist_regression.py).
Make sure that the package root is in your path when running the scripts or run them as modules, e.g.
```bash
/repa_vips/$ python3 -m experiments.BBVI.logistic_regression
```

## Citation
If you use this package in your own work please cite our paper:

Arenz, O.; Zhong, M.; Neumann, G. Efficient Gradient-Free Variational Inference using Policy Search. _Proceedings of the 35th International Conference on Machine Learning_. 2018.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

