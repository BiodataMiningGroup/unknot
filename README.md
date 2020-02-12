# UnKnoT

This repository contains the code for the paper "[Unsupervised knowledge transfer for object detection in marine environmental monitoring and exploration](#)."

## Requirements

The code is written in Python 3 and uses TensorFlow. It requires a GPU for decent processing speeds. It was tested with Ubuntu 18.04, Python 3.6 and an NVIDIA Tesla V100 GPU with 16 GB video memory.

Please refer to the [TensorFlow documentation](https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_101) on how to set up the GPU drivers and CUDA for TensorFlow.

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

Each dataset consists of the following files:

- `annotations/test.csv`: The BIIGLE CSV annotation report of the annotations of the test split of this dataset. These annotations are used to test the performance of the trained Mask R-CNN model.

- `annotations/train.csv`: The BIIGLE CSV annotation report of the annotations of the train split of this dataset. These annotations are used to generate the annotation patches which are transformed with scale transfer to be used to train the Mask R-CNN model.

- `images/`: Directory that contains all the original image files.

- `dataset.json`: JSON file that contains information about the dataset.
   - `name`: The name of the dataset.
   - `images_dir`: Name of the directory that contains the original image files.
   - `metadata_file`: Path to the CSV file that contains image metadata.
   - `test_annotations_file`: Path to the CSV file that contains the test annotations.
   - `train_annotations_file`: Path to the CSV file that contains the train annotations.
   - `annotation_patches_dir`: Name of the directory that should contain the scale-transferred annotation patches.
   - `crop_dimension`: Edge length of an annotation patch in pixels.

- `metadata.csv`: A CSV file that contains metadata for each original image file. In this case the distance of the camera to the sea floor is given for each image.

## Usage

Head over to the [example](example).

## Citation

Please cite our paper if it is helpful to your work:

```
TODO
```

## Acknowledgments

This work uses the code of the following repositories:

- [Mask R-CNN for Object Detection and Segmentation](https://github.com/matterport/Mask_RCNN)
