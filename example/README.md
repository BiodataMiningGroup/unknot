# UnKnoT Example

Please read the [paper](#) to learn more on how UnKnoT works. In the following, the practical application of the code is described.

**Important note:** The style transfer code uses PyTorch whereas the object detection code uses TensorFlow. These modules don't play well with each other when imported in the same script (they use mismatching cuDNN versions). Therefore, style transfer and object detection need to be executed with two separate scripts.

## 0. Preparation

1. Download two datasets, one as source dataset (e.g. S155) and one as target dataset (e.g. S083).

2. Set the `IMAGES_PER_GPU` in [`2_object_detection.py`](2_object_detection.py) to a number that fits to your GPU. As a rule of thumb, a GPU with 16 GB is able to handle five images of 512x512 px.

## 1. Scale- and style transfer

Run the following command to generate the scale- and style transferred annotation patches for the source dataset:

```bash
python3 1_scale_and_style_transfer.py <path to source dataset>/dataset.json <path to target dataset>/dataset.json
```

Style transfer takes about 25 minutes for one of the four provided datasets on an NVIDIA Tesla V100.

## 2. Object detection

Run the following command perform object detection on the target dataset based on training with the annotation patches generated in the previous step:

```bash
python3 2_object_detection.py <path to source dataset>/dataset.json <path to target dataset>/dataset.json results
```

Training takes about eight hours for one of the four provided datasets on an NVIDIA Tesla V100. The detection results will be stored in the `results` directory.

## 3. Evaluation

Run the following command perform the evaluation of an object detection result from the previous step (assuming the results were stored in the `results` directory):

```bash
python3 3_evaluation.py <path to target dataset>/dataset.json results
```

This will output an evaluation that includes the recall, precision, <i>F</i><sub>2</sub>-Score and <i>L</i>-Score of the object detection result.
