from concurrent.futures import ThreadPoolExecutor, as_completed
from pyvips import Image
from pyvips.error import Error as VipsError
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
import csv
import json
import numpy as np
import os
import warnings

class Dataset(object):
    def __init__(self, json_path, only_classes=[]):
        with open(json_path, 'r') as f:
            params = json.load(f)
        self.base_dir = os.path.dirname(json_path)
        self.images_dir = self.resolve_path(params['images_dir'])
        self.metadata_file = self.resolve_path(params['metadata_file'])
        self.test_annotations_file = self.resolve_path(params['test_annotations_file'])
        self.train_annotations_file = self.resolve_path(params['train_annotations_file'])
        self.training_images_path = self.resolve_path('training_images')
        self.coco_ann_file = self.resolve_path('training_annotations.json')
        self.only_classes = only_classes

    def resolve_path(self, path):
        return os.path.abspath(os.path.join(self.base_dir, path))

    def generate(self, scale_transfer_target, max_workers=4):
        if not os.path.exists(self.training_images_path):
           os.makedirs(self.training_images_path)

        target_scale = scale_transfer_target.get_mean_scale()
        metadata = self.get_image_metadata()

        executor = ThreadPoolExecutor(max_workers=max_workers)
        jobs = []

        for filename, annotations in self.get_training_annotations().items():
            scale_factor = metadata[filename] / target_scale
            jobs.append(executor.submit(self.process_image, filename, annotations, scale_factor))

        image_list = []

        for job in as_completed(jobs):
            a = job.result()
            if a is not False:
                image_list.append(a)

        annotation_list = []
        annotation_id = 0
        for image in image_list:
            annotations = image.pop('annotations')
            for annotation in annotations:
                annotation['id'] = annotation_id
                annotation_id += 1
                annotation_list.append(annotation)

        if len(image_list) == 0:
            raise Exception('No images in dataset. All corrupt?')

        # COCO format.
        # See: https://mmdetection.readthedocs.io/en/latest/user_guides/train.html#coco-annotation-format
        ann_file = {
            'images': image_list,
            'annotations': annotation_list,
            'categories': [{
                'id': 0,
                'name': 'interesting',
                'supercategory': 'interesting',
            }],
        }

        with open(self.coco_ann_file, 'w') as f:
            json.dump(ann_file, f)

    def should_ignore_label(self, label):
      return len(self.only_classes) > 0 and label not in self.only_classes

    def get_annotations(self, path):
        annotations = {}
        with open(path, 'r') as f:
            reader = csv.reader(f)
            # Ignore header.
            next(reader)
            for row in reader:
                if self.should_ignore_label(row[2]):
                    continue
                image_filename = row[8]
                center, radius = self.get_circle_from_row(row)
                if image_filename not in annotations:
                    annotations[image_filename] = []
                annotations[image_filename].append([center[0], center[1], radius])

        return annotations

    def get_training_annotations(self):
        return self.get_annotations(self.train_annotations_file)

    def get_test_annotations(self):
        return self.get_annotations(self.test_annotations_file)

    def get_image_metadata(self):
        metadata = {}
        with open(self.metadata_file, 'r') as f:
            reader = csv.reader(f)
            # Ignore header.
            next(reader)
            for row in reader:
                metadata[row[0]] = float(row[1])

        return metadata

    def get_mean_scale(self):
        metadata = self.get_image_metadata()

        return sum(metadata.values()) / len(metadata)

    def get_circle_from_row(self, row):
        points = np.array(row[13].strip('[]').split(','), dtype=float)
        shape_id = int(row[11])
        if shape_id == 1: # Point
            # Points get the default radius of 50
            return points[:2], 50
        elif shape_id == 4: # Circle
            return points[:2], points[2]
        elif points.size > 0:
            points = points.reshape(-1, 2)
            center = points.mean(axis=0)
            radius = np.NINF
            for c in points:
                radius = max(np.linalg.norm(center - c), radius)

            return center, radius
        else:
            raise Exception('Unsupported shape')

    def process_image(self, filename, annotations, scale_factor):
        try:
            source_path = os.path.join(self.images_dir, filename)
            target_path = os.path.join(self.training_images_path, filename)
            image = Image.new_from_file(source_path)
            if not os.path.exists(os.path.dirname(target_path)):
                os.makedirs(os.path.dirname(target_path))

            annotations = np.round(np.array(annotations, dtype=np.float32) * scale_factor)
            image = image.resize(scale_factor)
            image.write_to_file(target_path, strip=True, Q=95)

            coco_annotations = []

            for a in annotations:
                coco_annotations.append({
                    'id': 0, # Placeholder, will be updated to an unique ID later.
                    'image_id': filename,
                    'category_id': 0, # There is only one category.
                    'bbox': [
                        int(a[0] - a[2]), # px
                        int(a[1] - a[2]), # py
                        int(a[2] * 2), # width
                        int(a[2] * 2), # height
                    ],
                    'area': int((a[2] * 2)**2),
                    'iscrowd': 0,
                })


        except (IOError, OSError, VipsError) as e:
            warnings.warn(f'Image {filename} is corrupt or mising! Skipping...')

            return False

        return {
            'id': filename,
            'width': image.width,
            'height': image.height,
            'file_name': filename,
            'annotations': coco_annotations,
        }

    def evaluate_test_images(self, results_path):
        annotations = self.get_test_annotations()
        with open(results_path, 'r') as f:
            detections = json.load(f)

        total_annotations = 0
        total_detections = 0
        total_detected_annotations = 0
        recall = 0
        precision = 0

        for filename, anns in annotations.items():
            dets = detections.get(filename, [])
            if len(dets) == 0:
                TP = 0
            else:
                # Annotation matching developed by Torben Möller.
                mm = self.generate_maximum_matching(anns, dets)
                TP = np.count_nonzero(mm + 1) # unmatched vertices are represented by -1

            total_annotations += len(anns)
            total_detections += len(dets)
            total_detected_annotations += TP


        evaluation = {
            'total_annotations': total_annotations,
            'total_detections': total_detections,
            'total_detected_annotations': total_detected_annotations,
            'recall': total_detected_annotations / total_annotations,
            'precision': total_detected_annotations / total_detections,
        }

        return evaluation

    def generate_maximum_matching(gs, pr):
        adjacency = np.zeros((len(gs), len(pr)), dtype=bool)
        for i, gsa in enumerate(gs):
            for j, pra in enumerate(pr):
                adjacency[i, j] = annotations_match(gsa, pra)

        return maximum_bipartite_matching(csr_matrix(adjacency))

    def annotations_match(a, b):
        # Annotations match if one contains the center of the other.
        # Could also use IoU.
        distance = np.linalg.norm(a[:2] - b[:2])

        return distance <= a[2] or distance <= b[2]
