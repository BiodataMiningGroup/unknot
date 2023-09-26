# UnKnoT Example

Please read the [paper](https://doi.org/10.1109/ACCESS.2020.3014441) to learn more on how UnKnoT works. In the following, the practical application of the code is described.

## 0. Preparation

Download two datasets, one as source dataset (e.g. S155) and one as target dataset (e.g. S083).

## 1. Scale transfer

Run the following command to generate the scale transferred annotation patches for the source dataset:

```bash
python3 1_scale_transfer.py <path to source dataset>/dataset.json <path to target dataset>/dataset.json
```

## 2. Object detection

Run the following command perform object detection on the target dataset based on training with the annotation patches generated in the previous step:

```bash
python3 2_object_detection.py <path to source dataset>/dataset.json <path to target dataset>/dataset.json results
```

The detection results will be stored in `detections.json` in the `results` directory.

## 3. Evaluation

Run the following command perform the evaluation of an object detection result from the previous step (assuming the results were stored in the `results` directory):

```bash
python3 3_evaluation.py <path to target dataset>/dataset.json results/detections.json
```

This will output an evaluation that includes the recall and precision of the object detection result.
