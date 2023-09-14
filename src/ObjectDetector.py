from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.utils import setup_cache_size_limit_of_dynamo
from torch import cuda
from mmdet.apis import init_detector, inference_detector
import os

class ObjectDetector(object):
    def __init__(self, config_path, source, target, work_dir, workers=1, batch_size=16):
        self.config_path = config_path
        self.source = source
        self.target = target

        self.work_dir = os.path.abspath(work_dir)
        self.max_workers = workers

        self.cfg_options = {
            # Path to store the logfiles and final checkpoint to.
            'work_dir': self.work_dir,
            'train_dataloader': {
                # If multi-GPU training is implemented at some point, divide this by the
                # number of GPUs!
                'batch_size': batch_size,
                'num_workers': self.max_workers,
                'dataset': {
                    'ann_file': self.source.coco_ann_file,
                    'data_prefix': {
                        'img': self.source.images_dir,
                    },
                },
            },
            'test_dataloader': {
                'dataset': {
                    'ann_file': self.source.coco_ann_file,
                    'data_prefix': {
                        'img': self.source.images_dir,
                    },
                },
            },
            'test_evaluator': {
                'ann_file': self.source.coco_ann_file,
            },
            'classes': ('interesting', ),
            'gpu_ids': [0],
        }

    # Based on: https://github.com/open-mmlab/mmdetection/blob/master/tools/train.py
    def train(self):
        # Reduce the number of repeated compilations and improve
        # training speed.
        setup_cache_size_limit_of_dynamo()

        # load config
        cfg = Config.fromfile(self.config_path)

        cfg.merge_from_dict(self.cfg_options)

        if not os.path.exists(cfg.work_dir):
            os.makedirs(cfg.work_dir)

        # dump config
        # cfg.dump(os.path.join(cfg.work_dir, self.dump_config_name))

        runner = Runner.from_cfg(cfg)
        runner.train()

        return os.path.join(cfg.work_dir, 'epoch_12.pth')

    def detect(self, checkpoint_path):
        device = 'cuda:0' if cuda.is_available() else 'cpu'
        model = init_detector(self.config_path, checkpoint=checkpoint_path, device=device, cfg_options=self.cfg_options)

        images = self.target.get_image_metadata().keys()
        total_images = len(images)
        detections = {}

        for index, filename in enumerate(images):
            print(f'Image {index + 1} of {total_images} (#{image_id})')
            image_path = os.path.join(self.target.images_dir, filename)
            result = inference_detector(model, image_path)
            detections[filename] = self.process_result(image_id, result.pred_instances)

        with open(os.path.join(self.work_dir, 'detections.json'), 'w') as f:
            json.dump(detections, f)

    def process_result(self, image_id, pred):
        points = []
        for bbox, score, label in zip(pred.bboxes, pred.scores, pred.labels):
            x1, y1, x2, y2 = bbox.detach().cpu().numpy()
            r = round(max(x2 - x1, y2 - y1) / 2, 2)
            x = round((x1 + x2) / 2, 2)
            y = round((y1 + y2) / 2, 2)
            points.append([float(x), float(y), float(r), float(score)])

        return points
