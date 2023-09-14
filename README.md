# UnKnoT

This repository contains the code for the paper "[Unsupervised knowledge transfer for object detection in marine environmental monitoring and exploration](https://doi.org/10.1109/ACCESS.2020.3014441)."

The original code of the paper can be found as the [**paper**](https://github.com/BiodataMiningGroup/unknot/releases/tag/paper) release. The current state of the repository includes several updates and improvements.

## Requirements

The code is written in Python 3 and uses MMDetection. It requires a GPU for decent processing speeds. It was tested with Ubuntu 22.04, Python 3.10 and an NVIDIA Tesla V100 GPU with 16 GB video memory.

Please install the latest NVIDIA drivers, e.g.:

```
sudo apt install --no-install-recommends nvidia-driver-535-server
```

In addition, a few packages are required that can be installed in Ubuntu 22.04 with:

```
sudo apt install --no-install-recommends python3-pip libvips42
```

## Installation

Clone this repository with:

```
git clone https://github.com/BiodataMiningGroup/unknot.git
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
@article{zurowietz2020unsupervised,
  title={Unsupervised Knowledge Transfer for Object Detection in Marine Environmental Monitoring and Exploration}, 
  author={Zurowietz, Martin and Nattkemper, Tim Wilhelm},
  journal={IEEE Access}, 
  year={2020},
  doi={10.1109/ACCESS.2020.3014441}
}
```
