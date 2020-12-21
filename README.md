# SDLF: Simple Deep Learning Framework

A simple and generalized framework for deep learning projects.

# News

2020-12-21: SDLF v1.0 released!

2020-12-16: SDLF v0.9 released!

## Install on Ubuntu 16.04 / 18.04

### 1. Clone this repository

```bash
git clone https://github.com/wei-ForPrivateUse/sdlf.git
cd ./sdlf
```

### 2. Install dependencies

It is recommended to use Anaconda package manager.

```bash
conda install numpy tqdm
conda install -c conda-forge tensorboardX
```

### 3. Install Pytorch

This project is tested with pytorch v1.2.0, please follow the instructions in [Pytorch](https://pytorch.org/).

### 4. Build and setup this package

Build the package with ```bdist_wheel``` (do not use ```python setup.py install```).

```bash
python setup.py bdist_wheel
cd ./dist
```

Finally, use pip to install the generated whl file.

## Third-party libraries

This project uses torchplus, which is a part of SECOND.

* [SECOND](https://github.com/traveller59/second.pytorch): A 3D object detection algorithm.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details