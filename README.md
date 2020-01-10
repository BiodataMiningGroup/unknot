# UnKnoT

This repository contains the code for the paper "[Unsupervised knowledge transfer for object detection in marine environmental monitoring and exploration](#)."

## Requirements

The code is written in Python 3 and uses TensorFlow and PyTorch. It requires a GPU for decent processing speeds. It was tested with Ubuntu 18.04 and an NVIDIA Tesla V100 GPU with 16 GB video memory.

Please refer to the [TensorFlow documentation](https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_101) on how to set up the GPU drivers and CUDA for TensorFlow and PyTorch.

In addition, a few packages are required that can be installed in Ubuntu 18.04 with:

```
sudo apt install --no-install-recommends python3-pip python3-setuptools libvips42
```

## Installation

Clone this repository with:

```
git clone --recursive https://github.com/BiodataMiningGroup/unknot.git
```

Then run `pip3 install --user -r unknot/requirements.txt`. Alternatively you can use an Anaconda environment.

## Datasets

The datasets that were used in the paper are available at [Zenodo](https://zenodo.org/):

- S083: <https://doi.org/10.5281/zenodo.3600132>

- S155: <https://doi.org/10.5281/zenodo.3603803>

- S171: <https://doi.org/10.5281/zenodo.3603809>

- S233: <https://doi.org/10.5281/zenodo.3603815>

## Usage

Head over to the [example](example).

## Citation

Please cite our paper if it is helpful to your work:

```
TODO
```

## Acknowledgments

This work uses the code of the following repositories:

- [High-Resolution Network for Photorealistic Style Transfer](https://github.com/limingcv/Photorealistic-Style-Transfer)

- [Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN)
